#!/usr/bin/env python3
"""Resume-safe global-CLS / Evo-ViT KD study; logs every command and stage."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time
from evaluate_drop import ROOT


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def report(directory, state):
    lines = ["# Global CLS 및 Evo-ViT 실험", "", f"갱신(UTC): {now()}", "",
        f"상태: {state['status']}", "", "기준은 기존 전체 KD: Drop 50%·τ=1 Top-1 79.672%.",
        "공식 EMA에서 각각 새로 시작해 전체 backbone을 학습한다. ImageNet train 1,281,167장, 3에폭,",
        "batch 128 / micro 32, AdamW lr=5e-5, weight decay=0, warmup 500, cosine, seed 42, BF16.",
        "손실은 0.5 KL(3H)+0.5 KL(압축 6H), T=1. 유지율은 batch별 25/50%로 기존과 같다.", "",
        "- global_cls: 3H의 12블록 CLS attention을 decay=0.5 EMA로 누적하고 고정 Top-K로 6H를 실행한다.",
        "- evo_schedule: 같은 global CLS로 시작하고 6H에서 대표 토큰의 잔차를 비선택 토큰에 전파한다.",
        "  첫 2에폭은 블록별 재선택, 마지막 1에폭은 4블록마다 재선택한다. 추론은 4블록 단위다.",
        "- evo_layer 평가는 같은 evo_schedule 가중치에서 블록별 재선택을 적용하는 추론 비교다.",
        "  별도 학습된 layer-only 대조군이 아니므로 학습 스케줄 효과만을 분리하지 않는다.", "",
        "3H의 토큰 수는 유지한다. 압축은 재활용 6H 입력을 만든 뒤 시작한다. 대표 토큰 1개도",
        "기존 Drop과 같은 예산에 포함하므로 50%는 CLS 1+선택 패치 97+대표 1이다.",
        "논문 Eq.4–7 및 layer-to-stage(200/300 epoch)를 3에폭에 비례 적용한 ThinkingViT 변형이다.",
        "공식 DeiT 구현은 앞 4블록을 dense로 두고 placeholder를 그대로 보존하는 변형을 사용한다.",
        "이번 코드는 3H 정보를 활용해 6H 첫 블록부터 압축하며 논문의 대표 토큰 잔차를 갱신한다.",
        "논문의 전체 DeiT 학습 결과 재현으로 해석하지 않는다.", "",
        "평가: seed-42 validation 50,000장, FP32·TF32 허용, τ=1 고정. 임계값·epoch 선택 없음.",
        "기존 validation을 재사용하는 단일 seed 탐색이며 새 미사용 테스트 성능으로 보지 않는다.", "",
        "| 평가 | 표본 수 | 경로 | 전체 6H 경로 Top-1 | τ=1 Top-1 |", "| --- | ---: | --- | ---: | ---: |"]
    for summary_path in sorted(directory.glob("*/eval/summary.json")):
        result = json.loads(summary_path.read_text())
        for key, value in result["top1"].items():
            lines.append(f"| {summary_path.parent.parent.name} | {result['samples']} | {key} | {value['all']:.3f}% | {value['threshold_1']:.3f}% |")
        if "latency" in result:
            lines += ["", f"{summary_path.parent.parent.name} batch-1 지연(ms), 같은 모델의 τ=1·50% 유지·전처리/H2D 제외:",
                      "```json", json.dumps(result["latency"], ensure_ascii=False, indent=2), "```", ""]
    lines += ["", "## 실행 기록", "", "| 단계 | 시작 UTC | 종료 UTC | 상태 |", "| --- | --- | --- | --- |"]
    for stage in state["stages"]:
        lines.append(f"| {stage['name']} | {stage['started']} | {stage.get('finished', '')} | {stage['status']} |")
    lines += ["", f"원자료: `{directory.relative_to(ROOT)}`. 각 단계 명령·종료 코드: `study.json`.",
              "학습 진행: 각 `train/train_log.json`, 완료: `train/complete.json`. 체크포인트에 optimizer·RNG 포함.",
              "중단 재개 시 augmentation의 bit 단위 재현은 보장하지 않는다. Commit·push 없음.", "",
              "참고: [Evo-ViT 논문](https://arxiv.org/abs/2108.01390),",
              "[공식 구현](https://github.com/YifanXu74/Evo-ViT/blob/main/deit/evo_deit.py).", ""]
    (ROOT / "baseline/evolution_results.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/evo-vit-20261115")
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "runner.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = directory / "study.json"
    state = json.loads(path.read_text()) if path.exists() else {"started": now(), "stages": []}

    def execute(name, command, logfile):
        stage = {"name": name, "started": now(), "command": command, "status": "running"}
        state["stages"].append(stage)
        state["status"] = name
        atomic_json(path, state)
        report(directory, state)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with logfile.open("a") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            stage["pid"] = process.pid
            atomic_json(path, state)
            while process.poll() is None:
                report(directory, state)
                time.sleep(30)
        stage.update(finished=now(), exit_code=process.returncode,
                     status="complete" if process.returncode == 0 else "failed")
        atomic_json(path, state)
        report(directory, state)
        if process.returncode:
            raise RuntimeError(f"{name} failed: {logfile}")

    python = sys.executable
    def evaluate(name, weights, methods, ratios):
        destination = directory / name
        if (destination / "eval/summary.json").exists():
            return
        if (destination / "eval").exists():
            (destination / "eval").rename(destination / f"eval-incomplete-{time.time_ns()}")
        execute(f"{name}: validation", [python, "-u", "baseline/evaluate_evolution.py", "--weights", str(weights),
            "--methods", *methods, "--ratios", *map(str, ratios), "--latency-samples", "1000",
            "--output-dir", str(destination / "eval")], destination / "evaluate.log")

    try:
        evaluate("existing_kd", ROOT / "results/drop-ft-20261106/kd/train/student.pth",
                 ["last_cls", "global_cls", "evo_layer", "evo_stage"], [.5])
        for method in ("global_cls", "evo_schedule"):
            destination = directory / method
            if not (destination / "train/complete.json").exists():
                command = [python, "-u", "baseline/finetune_drop.py", "--token-method", method,
                    "--epochs", "3", "--lr", "5e-5", "--workers", "4", "--output-dir", str(destination / "train")]
                if (destination / "train/resume.pth").exists():
                    command += ["--resume", str(destination / "train/resume.pth")]
                execute(f"{method}: KD train", command, destination / "train.log")
            evaluate(method, destination / "train/student.pth",
                     ["last_cls", "global_cls"] if method == "global_cls" else ["evo_layer", "evo_stage"],
                     [.25, .5, .75])
        state["status"] = "모든 학습·평가 완료"
        state["finished"] = now()
    except Exception as error:
        state["status"] = f"실패: {error}"
        raise
    finally:
        atomic_json(path, state)
        report(directory, state)


if __name__ == "__main__":
    main()
