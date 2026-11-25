#!/usr/bin/env python3
"""Independent projection, Drop-6H norm, and Drop-6H LoRA accuracy experiments."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import torch
import numpy as np

from evaluate_drop import ROOT, load_model
from partial_tuning import MODES
from run_drop_ft_study import read_json, atomic_json

REPORT = ROOT / "baseline/partial_ft_results.md"
LEARNING_RATES = (5e-5, 2e-4, 1e-3)
COUNTS = {"p": 74113, "p_ln": 93313, "p_ln_lora": 240769}
SOURCE_FILES = ("baseline/run_partial_ft_study.py", "baseline/partial_tuning.py", "baseline/finetune_drop.py",
                "baseline/evaluate_drop.py", "baseline/evaluate_partial_calibration.py",
                "baseline/selective_refinement.py", "baseline/evaluate_h1_diagnosis.py",
                "third_party/ThinkingViT/timm/models/vision_transformer.py")


def duration(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 3600:02d}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def progress_json(path):
    # Workers may be replacing/truncating a progress file at this exact poll.
    # Only the report is allowed to defer it; experiment-state reads stay strict.
    try:
        return read_json(path)
    except (json.JSONDecodeError, FileNotFoundError):
        return None


def choose_learning_rate(candidates):
    valid = [row for row in candidates if row["kd_routed"] >= 0 and torch.isfinite(torch.tensor(row["kd_routed"]))]
    if not valid:
        raise ValueError("No finite calibration result")
    return min(valid, key=lambda row: (row["kd_routed"], row["lr"]))["lr"]


def compare_predictions(actual, control, original):
    route = actual["entropy"] >= 1
    pred = torch.where(route, actual["drop_50_correct"], actual["narrow_correct"])
    reference = torch.where(control["entropy"] >= 1, control["drop_50_correct"], control["narrow_correct"])
    difference = pred.to(torch.int8) - reference.to(torch.int8)
    better, worse = int((difference > 0).sum()), int((difference < 0).sum())
    n = len(pred)
    draw = np.random.default_rng(42).multinomial(n, [better/n, worse/n, 1-(better+worse)/n], size=20000)
    ci = np.quantile(100 * (draw[:, 0] - draw[:, 1]) / n, [.025, .975])
    original_route = original["entropy"] >= 1
    corrected = original_route & ~original["narrow_correct"] & original["wide_correct"]
    return {"drop50_gain_vs_full_kd_pp": 100 * (better-worse) / n, "better": better, "worse": worse,
            "paired_image_ci95_pp": ci.tolist(), "routed": int(route.sum()),
            "routed_top1": 100 * actual["drop_50_correct"][route].double().mean().item(),
            "teacher_corrected_images": int(corrected.sum()),
            "teacher_corrections_retained": int((corrected & actual["drop_50_correct"]).sum()),
            "note": "Image bootstrap conditional on fixed trained models, not training-seed variability."}


def update_report(state, output):
    now = datetime.now(timezone.utc)
    labels = {"p": "P: 재활용 변환", "p_ln": "P+LN", "p_ln_lora": "P+LN+LoRA"}
    lines = ["# 4번: 6H 부분 파인튜닝 결과", "",
             f"- 상태: **{state['status']}**", f"- GPU: {state['gpu']}; branch: `experiment/drop-ft`",
             f"- 갱신: {now.astimezone(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')}",
             f"- 원자료: `{output.relative_to(ROOT)}`. Commit·push 없음.", "",
             "## 방법", "",
             "세 방법을 공식 EMA에서 각각 새로 시작한다. 원본 backbone과 3H를 고정하고 재활용 projection/alpha, Drop 6H 전용 LayerNorm, Q/V LoRA(rank=alpha=8)를 차례로 추가한다.",
             "기존 대조군과 동일하게 train 전체 1,281,167장·3에폭·유효 batch 128·micro-batch 32·seed 42·AdamW(weight decay=0)·BF16을 사용한다.",
             "학습 유지율은 batch마다 25/50% 중 선택한다. 모든 블록에서 같은 토큰 집합을 사용하고, 주 평가는 고정 Drop 50%·τ=1이다. routing·선택기·블록별/이미지별 유지율은 바꾸지 않는다.",
             "손실은 기존 0.5 KL(3H)+0.5 KL(Drop 6H), T=1을 유지한다. 3H 보존 시 첫 항은 0이며 gradient가 없다.",
             "학습률은 validation을 사용하지 않고 train의 층화 pilot 20,000장(클래스당 20)·calibration 2,000장(클래스당 2, 서로 겹치지 않음)에서 선택한다.",
             "각 방법에 lr=5e-5/2e-4/1e-3을 같은 200 update·warmup 20·cosine 일정으로 시험하고, calibration routed Drop 50%의 Teacher KL이 가장 낮은 lr를 선택한다(동률이면 작은 lr).",
             "이후 선택한 lr로 공식 EMA부터 다시 시작해 전체 train 3에폭·warmup 500을 학습한다. 마지막 checkpoint를 평가하며 validation에서 best epoch를 고르지 않는다.",
             "최종 본 학습에는 calibration을 포함한 원래 train 전체를 사용한다. calibration은 학습률 선택용이며 최종 성능 평가용 데이터가 아니다.",
             "기존 전체 KD 기준 79.672%는 lr=5e-5의 별도 완료 실험이다. 새 방법은 train calibration으로 lr를 선택하므로 학습 파라미터 차이만의 순수 인과 비교로 주장하지 않는다.", "",
             "## 최종 정확도", "",
             "| 실험 | 학습 파라미터 | 선택 lr | Drop 25% τ=1 | Drop 50% τ=1 | Drop 75% τ=1 | 전체 6H |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
             "| 기존 전체 KD | 22,124,777 | 5e-5 | 78.388% | 79.672% | 79.978% | 81.092% |"]
    progress = []
    for mode in MODES:
        summary = progress_json(output / mode / "eval/summary.json")
        lr = state.get("selected_lr", {}).get(mode)
        if summary:
            top = summary["top1"]
            lines.append(f"| {labels[mode]} | {COUNTS[mode]:,} | {lr:g} | {top['drop_25_threshold_1']:.3f}% | "
                         f"{top['drop_50_threshold_1']:.3f}% | {top['drop_75_threshold_1']:.3f}% | {top['wide_all']:.3f}% |")
        else:
            lines.append(f"| {labels[mode]} | {COUNTS[mode]:,} | {lr if lr else '미선택'} | 미측정 | 미측정 | 미측정 | 미측정 |")
        config = progress_json(output / mode / "train/arguments.json")
        log = progress_json(output / mode / "train/train_log.json")
        if log and config:
            last = log[-1]
            remaining = (config["steps"] - last["step"]) * config["batch_size"] / max(last["images_per_s"], 1)
            progress.append(f"- {labels[mode]}: {last['step']:,}/{config['steps']:,} update, "
                            f"누적 {last['images_per_s']:.1f} images/s, 현재 속도 기준 잔여 {duration(remaining)}.")
    if progress:
        lines += ["", "## 본 학습 진행", "", *progress]
    lines += ["", "## 3H 보존 및 메모리", "",
              "매 checkpoint에서 동결 파라미터 hash와 고정 입력 4장의 3H logits·attention·entropy·routing·25/50/75% Top-K 동일성을 검사한다.",
              "최종 전체 validation에서는 저장된 원본 entropy·3H 정오답과 이미지별로 비교한다. full 6H는 projection 변경의 영향을 받을 수 있다.", "",
              "| 실험 | 동결 파라미터 유지 | 3H guard max logits 차이 | 전체 validation entropy 변화 | peak allocated MiB |",
              "| --- | --- | ---: | ---: | ---: |"]
    for mode in MODES:
        guard = progress_json(output / mode / "train/preservation.json")
        done = progress_json(output / mode / "train/complete.json")
        validation = progress_json(output / mode / "validation_preservation.json")
        if guard:
            lines.append(f"| {labels[mode]} | {guard['frozen_parameters_unchanged']} | {guard['logits_max_abs_diff']} | "
                         f"{validation['entropy_max_abs_diff'] if validation else '미측정'} | {done['peak_allocated_mib'] if done else '진행 중'} |")
    lines += ["", "## 기존 전체 KD와 이미지별 비교", "",
              "| 실험 | Drop 50% 차이(%p) | paired 95% 구간 | routed Top-1 | 원본 교정 유지 |",
              "| --- | ---: | --- | ---: | --- |"]
    for mode in MODES:
        comparison = progress_json(output / mode / "comparison.json")
        if comparison:
            ci = comparison["paired_image_ci95_pp"]
            lines.append(f"| {labels[mode]} | {comparison['drop50_gain_vs_full_kd_pp']:+.3f} | {ci[0]:+.3f}~{ci[1]:+.3f} | "
                         f"{comparison['routed_top1']:.3f}% | {comparison['teacher_corrections_retained']}/{comparison['teacher_corrected_images']} |")
    lines += ["", "## Pilot (train calibration, validation 정확도 아님)", "",
              "| 방법 | lr | routed KL | calibration Top-1 |", "| --- | ---: | ---: | ---: |"]
    for mode, entries in state.get("pilots", {}).items():
        for row in entries:
            lines.append(f"| {mode} | {row['lr']:g} | {row['kd_routed']:.6f} | {row['top1']:.3f}% |")
    lines += ["", "## 시간 (KST, 시:분:초)", "",
              "기존에 준비한 데이터 다운로드 시간은 18:49:41이며 이번 소요 시간에는 중복 합산하지 않는다. 이전 결과는 drop_ft_experiments.md 참조.", "",
              "| 단계 | 시작 | 종료 | 소요 | 상태 |", "| --- | --- | --- | ---: | --- |"]
    for record in state.get("commands", []):
        def clock(value):
            return datetime.fromisoformat(value).astimezone(ZoneInfo("Asia/Seoul")).strftime("%m-%d %H:%M:%S") if value else "진행 중"
        elapsed = record.get("elapsed_seconds", (now - datetime.fromisoformat(record["started"])).total_seconds())
        lines.append(f"| {record['label']} | {clock(record['started'])} | {clock(record.get('finished'))} | "
                     f"{duration(elapsed)} | {record.get('outcome', '진행 중')} |")
    finish = datetime.fromisoformat(state["finished"]) if state.get("finished") else now
    lines += ["", f"전체 경과: {duration((finish-datetime.fromisoformat(state['started'])).total_seconds())}",
              "", "단일 seed의 비교다. pilot과 본 학습은 별도 초기화하며, 작은 정확도 차이는 이미지별 paired 구간과 추가 seed 검증을 구분해서 해석한다.",
              "이번 단계는 정확도 실험이며 GPU runtime 최적화는 재실행하지 않는다. LoRA는 추가 추론 연산·저장량이 있으므로 추론 GMACs가 같다고 주장하지 않는다.",
              "중단 후 optimizer·scheduler·RNG와 batch 위치를 복원하지만 worker augmentation의 bit 단위 재현은 보장하지 않는다."]
    if state.get("error"):
        lines += ["", f"오류: `{state['error']}`"]
    temporary = REPORT.with_suffix(".tmp.md")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(REPORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/partial-ft-20261109")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    state_file = output / "study.json"
    state = read_json(state_file) or {"started": datetime.now(timezone.utc).isoformat(),
                                     "gpu": torch.cuda.get_device_name(), "status": "준비 중", "commands": [],
                                     "pilots": {}, "selected_lr": {}}
    python = str(ROOT / ".venv/bin/python")
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    if state.get("source_sha256") and state["source_sha256"] != hashes:
        raise ValueError("Study source changed; use a new output directory")
    state["source_sha256"] = hashes
    state["environment"] = {"torch": str(torch.__version__), "cuda": torch.version.cuda, "python": sys.version}
    state["checkpoint_sha256"] = hashlib.sha256((ROOT / "checkpoints/thinkingvit_3h6h_ema.pth").read_bytes()).hexdigest()
    state["preflight"] = read_json(output / "preflight.json")
    state.pop("error", None)

    def persist():
        atomic_json(state_file, state)
        update_report(state, output)

    def status(value):
        state["status"] = value
        print(datetime.now().astimezone().isoformat(timespec="seconds"), value, flush=True)
        persist()

    def execute(command, logfile, label):
        for name, digest in hashes.items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Source changed while study was running: {name}")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        record = {"label": label, "command": command, "log": str(logfile),
                  "started": datetime.now(timezone.utc).isoformat()}
        state["commands"].append(record)
        persist()
        start = time.perf_counter()
        try:
            with logfile.open("a") as stream:
                process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                record["pid"] = process.pid
                persist()
                while True:
                    try:
                        process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        record["last_seen"] = datetime.now(timezone.utc).isoformat()
                        persist()
                record["returncode"] = process.returncode
                record["outcome"] = "완료" if process.returncode == 0 else "실패"
                if process.returncode:
                    raise subprocess.CalledProcessError(process.returncode, command)
        except BaseException as error:
            record.setdefault("outcome", f"중단: {type(error).__name__}")
            raise
        finally:
            record["finished"] = datetime.now(timezone.utc).isoformat()
            record["elapsed_seconds"] = time.perf_counter() - start
            persist()

    def train(mode, destination, lr, pilot=False):
        if (destination / "complete.json").exists():
            return
        command = [python, "-u", "baseline/finetune_drop.py", "--partial-mode", mode,
                   "--epochs", "3", "--lr", str(lr), "--workers", "4", "--batch-size", "128",
                   "--micro-batch-size", "32", "--output-dir", str(destination)]
        if pilot:
            command += ["--max-steps", "200", "--warmup", "20", "--train-indices", str(output / "pilot_indices.pt")]
        if (destination / "resume.pth").exists():
            command += ["--resume", str(destination / "resume.pth")]
        execute(command, destination.parent / "train.log", f"{mode} {'pilot' if pilot else '본 학습'} lr={lr:g}")

    try:
        for record in state["commands"]:
            if not record.get("finished"):
                pid = record.get("pid")
                if pid and Path(f"/proc/{pid}").exists():
                    raise RuntimeError(f"Previous worker may still be running: {pid}")
                record["finished"] = record.get("last_seen", record["started"])
                record["elapsed_seconds"] = (datetime.fromisoformat(record["finished"]) - datetime.fromisoformat(record["started"])).total_seconds()
                record["outcome"] = "중단; 종료 시각은 마지막 관측값"
        status("train pilot/calibration 분리 준비")
        if not (output / "pilot_indices.pt").exists():
            from torchvision.datasets import ImageFolder
            dataset = ImageFolder(ROOT / "data/imagenet1k/train")
            if len(dataset) != 1281167:
                raise ValueError("Complete train is required")
            classes = [[] for _ in range(1000)]
            for index, target in enumerate(dataset.targets):
                classes[target].append(index)
            rng = random.Random(20261109)
            training, calibration = [], []
            for values in classes:
                rng.shuffle(values)
                training.extend(values[:20]); calibration.extend(values[20:22])
            if len(training) != 20000 or len(calibration) != 2000 or set(training) & set(calibration):
                raise AssertionError("Invalid stratified train split")
            rng.shuffle(training); rng.shuffle(calibration)
            torch.save(training, output / "pilot_indices.pt")
            torch.save(calibration, output / "calibration_indices.pt")
            atomic_json(output / "split.json", {"split": "train", "seed": 20261109,
                                                "pilot": len(training), "calibration": len(calibration),
                                                "disjoint": True, "pilot_per_class": 20, "calibration_per_class": 2})
            del dataset, classes
        for mode in MODES:
            candidates = []
            for lr in LEARNING_RATES:
                directory = output / "pilots" / mode / f"lr{lr:g}"
                status(f"{mode}: 학습률 pilot lr={lr:g}")
                train(mode, directory / "train", lr, pilot=True)
                metric = directory / "calibration.json"
                if not metric.exists():
                    execute([python, "-u", "baseline/evaluate_partial_calibration.py", "--weights", str(directory / "train/student.pth"),
                             "--indices", str(output / "calibration_indices.pt"), "--output", str(metric)],
                            directory / "calibration.log", f"{mode} train calibration lr={lr:g}")
                candidates.append({"lr": lr, **read_json(metric)})
                state["pilots"][mode] = candidates
                persist()
            selected = choose_learning_rate(candidates)
            state["selected_lr"][mode] = selected
            status(f"{mode}: 본 학습, lr={selected:g}")
            train(mode, output / mode / "train", selected)
            evaluation = output / mode / "eval"
            if not (evaluation / "summary.json").exists():
                if evaluation.exists():
                    evaluation.rename(evaluation.with_name(f"eval-incomplete-{time.time_ns()}"))
                status(f"{mode}: 전체 validation 평가")
                execute([python, "-u", "baseline/evaluate_drop.py", "--weights", str(output / mode / "train/student.pth"),
                         "--workers", "4", "--output-dir", str(evaluation)], output / mode / "evaluate.log", f"{mode} validation 50,000장")
            original = torch.load(ROOT / "results/20261106T124010.390968Z-drop-eval-official/per_image.pt", map_location="cpu", weights_only=True)
            actual = torch.load(evaluation / "per_image.pt", map_location="cpu", weights_only=True)
            guard = {"samples": len(actual["entropy"]), "entropy_max_abs_diff": (actual["entropy"] - original["entropy"]).abs().max().item(),
                     "narrow_correctness_mismatches": int((actual["narrow_correct"] != original["narrow_correct"]).sum()),
                     "routing_mismatches": int(((actual["entropy"] >= 1) != (original["entropy"] >= 1)).sum())}
            atomic_json(output / mode / "validation_preservation.json", guard)
            if guard["samples"] != 50000 or guard["entropy_max_abs_diff"] != 0 or guard["narrow_correctness_mismatches"] or guard["routing_mismatches"]:
                raise AssertionError("3H changed on validation")
            control = torch.load(ROOT / "results/drop-ft-20261106/kd/eval/per_image.pt", map_location="cpu", weights_only=True)
            atomic_json(output / mode / "comparison.json", compare_predictions(actual, control, original))
            status(f"{mode}: 학습·평가 완료")
        state["finished"] = datetime.now(timezone.utc).isoformat()
        status("세 실험 완료")
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
        status("실행 중단 — 오류 확인 필요")
        raise


if __name__ == "__main__":
    main()
