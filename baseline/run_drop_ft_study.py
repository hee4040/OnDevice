#!/usr/bin/env python3
"""Run the local KD reproduction, independent ablations, and a selected combination.

All runs start at the official checkpoint. Selection uses only the first 25,000
validation images in evaluate_drop's fixed seed-42 order; the other half is held out.
This process writes its state/report after every stage and supports restarting.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import torch

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "baseline/drop_ft_experiments.md"
REFERENCE = ROOT / "baseline/reports/2026-11-03"
BASELINE_NAMES = ("kd", "ce", "full", "feature")
OPTIONS = {"kd": {}, "ce": {"ce": 1.0}, "full": {"full": 0.5}, "feature": {"feature": 1.0}}


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def half_scores(per_image):
    """Use the existing image order, identical across runs, without rerouting bias."""
    routed = per_image["entropy"] >= 1.0
    narrow = per_image["narrow_correct"]
    drop = torch.where(routed, per_image["drop_50_correct"], narrow)
    full = per_image["wide_correct"]
    mixed_full = torch.where(routed, full, narrow)
    result = {}
    for name, selection in (("selection", slice(0, 25000)), ("held_out", slice(25000, 50000))):
        result[name] = {key: 100 * value[selection].double().mean().item()
                        for key, value in (("drop_50", drop), ("full_6h", full), ("full_threshold_1", mixed_full))}
    return result


def choose_components(scores):
    """Predeclared practical thresholds in percentage points; no held-out access."""
    control = scores["kd"]["selection"]
    selected = []
    for name in ("ce", "full", "feature"):
        current = scores[name]["selection"]
        drop_gain = current["drop_50"] - control["drop_50"]
        full_gain = current["full_6h"] - control["full_6h"]
        # Keep accuracy improvements without a large full-path regression, or
        # full-path preservation without a material loss on the target drop path.
        if ((drop_gain >= .10 and full_gain >= -.15)
                or (name == "full" and full_gain >= .10 and drop_gain >= -.10)):
            selected.append(name)
    return selected


def timing_report(state, now=None):
    """Wall times include model/data-loader setup; overlapping intervals count once."""
    now = now or datetime.now(timezone.utc)
    end = datetime.fromisoformat(state["completed_at"]) if state.get("completed_at") else now
    data = state.get("data_preparation")
    rows = []
    if data:
        rows.append(("train 다운로드·추출", data))
    for command in state.get("commands", []):
        path = Path(command["log"])
        experiment = path.parent.name
        kind = {"train.log": "학습", "evaluate.log": "정확도 평가", "latency.log": "추론 시간 측정",
                "prepare.log": "데이터 준비(재시도 포함)"}.get(path.name, path.stem)
        rows.append((f"{experiment}: {kind}", command))
    if not rows:
        return ["", "## 소요 시간", "", "시간 기록 대기 중."]

    def clock(value):
        if not value:
            return "진행 중"
        return datetime.fromisoformat(value).astimezone(ZoneInfo("Asia/Seoul")).strftime("%m-%d %H:%M:%S")

    def duration(seconds):
        seconds = max(0, round(seconds))
        return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"

    lines = ["", "## 소요 시간", "",
             "시각은 한국 시간(KST), 소요 시간은 시:분:초다. 학습·평가 시간에는 해당 프로세스의 모델·로더 준비와 저장 시간이 포함된다.",
             "데이터는 다운로드와 이미지 추출이 번갈아 실행되어 두 작업을 합쳐 기록한다. 재시도 시간도 포함하며, 이미 준비돼 있던 validation은 제외한다.", "",
             "| 단계 | 시작(KST) | 종료(KST) | 소요 시간 | 상태 |",
             "| --- | --- | --- | ---: | --- |"]
    intervals = []
    for label, record in rows:
        start = datetime.fromisoformat(record["started"])
        finish = datetime.fromisoformat(record["finished"]) if record.get("finished") else end
        seconds = record.get("elapsed_seconds", (finish - start).total_seconds())
        status = record.get("outcome", "완료" if record.get("finished") else "진행 중")
        lines.append(f"| {label} | {clock(record['started'])} | {clock(record.get('finished'))} | {duration(seconds)} | {status} |")
        intervals.append((start, finish))
    intervals.sort()
    first = intervals[0][0]
    active = 0.0
    begin, finish = intervals[0]
    for next_begin, next_finish in intervals[1:]:
        if next_begin <= finish:
            finish = max(finish, next_finish)
        else:
            active += (finish - begin).total_seconds()
            begin, finish = next_begin, next_finish
    active += (finish - begin).total_seconds()
    total = max(0, (end - first).total_seconds())
    lines += ["", f"- 전체 경과 시간(위 기록의 첫 단계 시작부터): **{duration(total)}**" + (" (진행 중)" if not state.get("completed_at") else ""),
              f"- 기록된 단계 사이 대기·중단·정리 시간: **{duration(total - active)}**. 겹치는 작업 시간은 중복 합산하지 않는다.",
              "- 중단된 시도도 별도 행으로 보존한다. 다운로드 시작은 실행 중 프로세스의 시작 시각(초 단위), 종료는 전체 데이터 검증 완료 파일의 저장 시각으로 기록한다.",
              "- 시간 기록을 추가하기 전에 완료한 원본 정확도 평가·원본 GPU 지연 측정·구현 점검은 위 합계에서 제외한다."]
    return lines


def update_report(state, directory):
    local = read_json(Path(state["official_eval"]) / "summary.json")
    ref = read_json(REFERENCE / "drop_eval_official.json")
    lines = ["# Drop fine-tuning 실험", "",
             f"- 브랜치: `experiment/drop-ft`, GPU: {state['gpu']}",
             f"- 상태: **{state['status']}** (갱신: {datetime.now().astimezone().isoformat(timespec='seconds')})",
             f"- 전체 실행 기록: `{directory.relative_to(ROOT)}`; GitHub push 없음.", "",
             "## 방법", "",
             "모든 학습은 공식 EMA에서 각각 새로 시작한다. ImageNet train 1,281,167장만 학습하고 validation 50,000장으로 평가한다.",
             "3에폭, 유효 batch 128(micro-batch 32의 gradient accumulation), AdamW lr 5e-5, weight decay 0, warmup 500, cosine decay, seed 42, BF16.",
             "유지 비율은 기존 실험과 같이 매 batch 25%/50% 중 선택한다. 주 지표는 Drop 50%, entropy τ=1의 Top-1이다.",
             "검증은 기존 코드와 같은 FP32·TF32 허용, batch 64다. 3H/전체 6H/Drop 25·50·75%를 함께 확인한다.", "",
             "| 실험 | 기존 KD에 추가하는 손실 |",
             "| --- | --- |",
             "| KD 재현 | 0.5 KL(3H) + 0.5 KL(Drop 6H), T=1 |",
             "| CE | + 0.5 CE(3H, 정답) + 0.5 CE(Drop 6H, 정답) |",
             "| Full 보존 | + 0.5 KL(Student 전체 6H, Teacher 전체 6H) |",
             "| Feature | + 1.0 × 4·8·12번째 6H 블록의 토큰 특징 MSE 평균 |", "",
             "Feature는 남긴 패치의 **원래 위치**와 CLS를 Teacher 전체 토큰에 대응시키고, 각 토큰에 파라미터 없는 layer norm을 적용한 뒤 비교한다. Teacher에는 gradient를 주지 않는다.",
             "처음부터 전체 정답·출력·특징을 함께 바꾸지 않고, 세 실험을 KD 대조군과 독립적으로 비교한다.", "",
             "조합 규칙은 실행 전에 고정했다. seed-42 validation 순서의 앞 25,000장에서 Drop 50%가 KD 대비 ≥0.10%p 개선되고 전체 6H 하락이 ≤0.15%p이면 채택한다.",
             "Full 보존은 전체 6H가 ≥0.10%p 개선되고 Drop 50% 하락이 ≤0.10%p여도 채택한다. 2개 이상 채택되면 같은 가중치로 조합을 공식 EMA부터 새로 학습한다.",
             "뒤 25,000장은 조합 선택에 사용하지 않고 최종 확인용으로 남긴다. 전체 50,000장 수치는 기존 보고와의 비교용이며 선택용 데이터도 포함한다.", "",
             "## 4090 대비 원본 재현 (전체 validation)", "",
             "| 경로 | 4090 Top-1 | 현재 GPU Top-1 | 차이(%p) |", "| --- | ---: | ---: | ---: |"]
    if local:
        for label, key in (("3H", "narrow_all"), ("전체 6H", "wide_all"), ("원본 τ=1", "wide_threshold_1"),
                           ("학습 전 Drop 25%", "drop_25_threshold_1"), ("학습 전 Drop 50%", "drop_50_threshold_1"),
                           ("학습 전 Drop 75%", "drop_75_threshold_1")):
            old, new = ref["top1"][key], local["top1"][key]
            lines.append(f"| {label} | {old:.3f}% | {new:.3f}% | {new-old:+.3f} |")
    lines += ["", "## 파인튜닝 결과 (전체 validation)", "",
              "| 실험 | Drop 25% τ=1 | Drop 50% τ=1 | Drop 75% τ=1 | 전체 6H | Drop 50%의 로컬 KD 대비 차이 |",
              "| --- | ---: | ---: | ---: | ---: | ---: |",
              "| 기존 4090 KD 기록 | 78.430% | 79.650% | 79.990% | 81.032% | 참고값 |"]
    baseline = read_json(directory / "kd/eval/summary.json")
    for name in (*BASELINE_NAMES, "combined"):
        row = read_json(directory / name / "eval/summary.json")
        if row:
            top = row["top1"]
            gain = top["drop_50_threshold_1"] - baseline["top1"]["drop_50_threshold_1"] if baseline else None
            change = f"{gain:+.3f}%p" if gain is not None else "—"
            lines.append(f"| {name} | {top['drop_25_threshold_1']:.3f}% | {top['drop_50_threshold_1']:.3f}% | "
                         f"{top['drop_75_threshold_1']:.3f}% | {top['wide_all']:.3f}% | {change} |")
        elif name != "combined" or state.get("components"):
            lines.append(f"| {name} | 미측정 | 미측정 | 미측정 | 미측정 | — |")
    lines += ["", "## 조합 선택 및 최종 확인", ""]
    if "components" in state:
        lines.append("채택: " + (", ".join(state["components"]) or "없음") + ".")
        if len(state["components"]) < 2:
            lines.append("조건을 만족한 항목이 2개 미만이어서 중복되거나 근거 없는 조합 학습은 생략한다.")
    else:
        lines.append("개별 실험 완료 후 앞 25,000장 결과로 결정한다. 현재 조합 효과는 아직 측정하지 않았다.")
    scores = state.get("scores", {})
    if scores:
        lines += ["", "| 실험 | 선택 25,000장 Drop 50% | 최종 확인 25,000장 Drop 50% | 최종 확인 전체 6H |",
                  "| --- | ---: | ---: | ---: |"]
        for name, score in scores.items():
            lines.append(f"| {name} | {score['selection']['drop_50']:.3f}% | {score['held_out']['drop_50']:.3f}% | "
                         f"{score['held_out']['full_6h']:.3f}% |")
    lines += ["", "## GPU 시간 비교", "",
              "같은 validation 1,000장, batch 1, τ=1, synchronized wall-clock 평균. 전처리·H2D 제외, 분기 동기화 포함. 로컬은 방법 순서를 섞어 3회 반복하고 평균 시간의 중앙값(최소~최대)을 표시한다. 4090은 기존 1회 기록이다. 파인튜닝 가중치 측정은 학습 후 별도로 수행한다.", "",
              "| 경로 | 4090 원본 가중치 | 현재 GPU 원본 가중치 | 현재 GPU KD 가중치 |",
              "| --- | ---: | ---: | ---: |"]
    old_latency = read_json(REFERENCE / "pipeline_latency.json")
    latency = read_json(directory / "official-latency/summary.json")
    kd_latency = read_json(directory / "kd/latency/summary.json")
    for name in ("Official, threshold 1.0", "Drop Top-K 25%", "Drop Top-K 50%", "Drop Top-K 75%"):
        def milliseconds(data):
            if data:
                values = [row["mean_ms"] for row in data["rows"] if row["setting"] == "GPU" and row["method"] == name]
                if values:
                    spread = f" ({min(values):.3f}~{max(values):.3f})" if len(values) > 1 else ""
                    return f"{statistics.median(values):.3f} ms{spread}"
            return "미측정"
        lines.append(f"| {name} | {milliseconds(old_latency)} | {milliseconds(latency)} | {milliseconds(kd_latency)} |")
    lines += timing_report(state)
    lines += ["", "## 해석 시 주의", "",
              "- 구현 검증: 단위 테스트 32개 통과. 임의 텐서로 KD·CE·Full·Feature·조합의 실제 GPU 역전파를 점검했으며 최대 allocated memory는 약 2.3GiB였다. 이는 학습 정확도 결과가 아니다.",
              "- 4090 파인튜닝 가중치는 이 컴퓨터에 없다. 원본 가중치의 GPU 간 평가 비교와, 로컬에서 다시 학습한 KD의 재현 비교는 서로 다르다.",
              "- 원본 재현은 Top-1 집계 비교다. 4090의 이미지별 예측 원자료가 없어 모든 이미지 예측의 동일성을 검증한 것은 아니다.",
              "- 4090 학습은 20,000 step에서 중단 후 AdamW 상태를 초기화해 재개했다. 로컬 학습은 연속 실행하며 micro-batch도 다르므로 재학습 차이를 GPU 차이만으로 해석하지 않는다.",
              "- 재개 파일에는 모델·optimizer·scheduler·RNG 상태를 저장한다. DataLoader worker의 augmentation 상태는 저장하지 않아 재부팅 후 완전한 bit 단위 재현을 보장하지 않는다.",
              "- 단일 seed 실험이다. 소폭 개선은 추가 seed 검증 전에는 확정적 개선으로 해석하지 않는다.",
              "- 다운로드한 train 원본 이미지는 유지하고, 전용 임시 Parquet 캐시만 shard별 추출·manifest 저장 후 제거한다.", ""]
    if state.get("error"):
        lines += [f"실행 오류: `{state['error']}`. 원인을 확인하고 같은 명령으로 재개할 수 있다.", ""]
    REPORT.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/drop-ft-20261106")
    parser.add_argument("--official-eval", type=Path, required=True)
    parser.add_argument("--data-pid", type=int, help="Wait for an already running preparation process")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock_file = (directory / "runner.lock").open("w")
    import fcntl
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    state_file = directory / "study.json"
    state = read_json(state_file) or {"official_eval": str(args.official_eval.resolve()),
                                     "gpu": torch.cuda.get_device_name(), "scores": {}, "commands": []}
    state["environment"] = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                            "python": sys.version, "gpu": torch.cuda.get_device_name()}
    python = str(ROOT / ".venv/bin/python")

    def status(message):
        state["status"] = message
        if message == "모든 실험 완료":
            state["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_json(state_file, state)
        update_report(state, directory)
        print(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}", flush=True)

    def execute(command, logfile):
        logfile.parent.mkdir(parents=True, exist_ok=True)
        record = {"command": command, "log": str(logfile),
                  "started": datetime.now(timezone.utc).isoformat()}
        state["commands"].append(record)
        atomic_json(state_file, state)
        update_report(state, directory)
        start = time.perf_counter()
        try:
            with logfile.open("a") as output:
                result = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
            record["returncode"] = result.returncode
            record["outcome"] = "완료" if result.returncode == 0 else "실패"
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command)
        except BaseException as error:
            record.setdefault("outcome", f"중단: {type(error).__name__}")
            raise
        finally:
            record["finished"] = datetime.now(timezone.utc).isoformat()
            record["elapsed_seconds"] = time.perf_counter() - start
            atomic_json(state_file, state)
            update_report(state, directory)

    def experiment(name, options):
        destination = directory / name
        if not (destination / "train/complete.json").exists():
            status(f"{name}: 학습 중")
            command = [python, "-u", "baseline/finetune_drop.py", "--epochs", "3", "--lr", "5e-5",
                       "--batch-size", "128", "--micro-batch-size", "32", "--workers", "4",
                       "--output-dir", str(destination / "train")]
            for option, weight in options.items():
                command += [f"--{option}-weight", str(weight)]
            resume = destination / "train/resume.pth"
            if resume.exists():
                command += ["--resume", str(resume)]
            execute(command, destination / "train.log")
        evaluation = destination / "eval"
        if not (evaluation / "summary.json").exists():
            # A crashed evaluation has no summary; preserve it and run into a new directory.
            if evaluation.exists():
                evaluation.rename(destination / f"eval-incomplete-{time.time_ns()}")
            status(f"{name}: validation 평가 중")
            execute([python, "-u", "baseline/evaluate_drop.py", "--weights", str(destination / "train/student.pth"),
                     "--workers", "4", "--output-dir", str(evaluation)], destination / "evaluate.log")
        per_image = torch.load(evaluation / "per_image.pt", map_location="cpu", weights_only=True)
        if len(per_image["entropy"]) != 50000:
            raise ValueError("Incomplete validation cannot be used for selection")
        state["scores"][name] = half_scores(per_image)
        status(f"{name}: 학습·정확도 평가 완료")
        if name == "kd" and not (destination / "latency/summary.json").exists():
            execute([python, "-u", "baseline/measure_latency.py", "--mode", "pipeline", "--images", "1000",
                     "--gpu-only", "--repeats", "3", "--weights", str(destination / "train/student.pth"),
                     "--output-dir", str(destination / "latency")], destination / "latency.log")

    try:
        hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in (
            "baseline/finetune_drop.py", "baseline/drop_ft_losses.py", "baseline/evaluate_drop.py",
            "baseline/selective_refinement.py", "baseline/evaluate_h1_diagnosis.py", "baseline/measure_latency.py",
            "baseline/run_drop_ft_study.py", "scripts/prepare_imagenet.py")}
        if state.get("source_sha256") and hashes != state["source_sha256"]:
            raise ValueError("Study source changed; use a new output directory for comparable runs")
        state["source_sha256"] = hashes
        state.pop("error", None)
        ready = ROOT / "data/imagenet1k/.preparation/train-complete.json"
        if not ready.exists() and args.data_pid:
            status("train 다운로드·추출 대기 중 (학습 미시작)")
            last_prepared = -1
            while not ready.exists():
                try:
                    os.kill(args.data_pid, 0)
                except ProcessLookupError:
                    break
                prepared = len(list(ready.parent.glob("train-*-of-*.json")))
                if prepared != last_prepared:
                    status(f"train 다운로드·추출 중 ({prepared}/294 shards; 학습 미시작)")
                    last_prepared = prepared
                time.sleep(30)
        if not ready.exists():
            status("train 다운로드·추출 중 (학습 미시작)")
            execute([python, "-u", "scripts/prepare_imagenet.py", "--splits", "train", "--discard-shards"],
                    directory / "prepare.log")
        # Freeze provenance before training. The train script verifies image counts too.
        if state.get("data_preparation") and not state["data_preparation"].get("finished"):
            timing = state["data_preparation"]
            finished = datetime.fromtimestamp(ready.stat().st_mtime, timezone.utc)
            timing["finished"] = finished.isoformat()
            timing["elapsed_seconds"] = (finished - datetime.fromisoformat(timing["started"])).total_seconds()
            timing["outcome"] = "완료"
        state["train_data"] = read_json(ready)
        state["checkpoint_sha256"] = hashlib.sha256((ROOT / "checkpoints/thinkingvit_3h6h_ema.pth").read_bytes()).hexdigest()
        for name in BASELINE_NAMES:
            experiment(name, OPTIONS[name])
        state["components"] = choose_components(state["scores"])
        status("개별 실험 완료, 선택용 validation으로 조합 결정")
        if len(state["components"]) >= 2:
            options = {key: value for name in state["components"] for key, value in OPTIONS[name].items()}
            experiment("combined", options)
        status("모든 실험 완료")
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
        status("실행 중단 — 오류 확인 필요")
        raise


if __name__ == "__main__":
    main()
