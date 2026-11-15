#!/usr/bin/env python3
"""Queue Global-CLS -> slow-fast training after the active original study."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from evolution_report import ROOT, read, write_report


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/evo-vit-20261115")
    args = parser.parse_args()
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    own_lock = (directory / "warm_start_runner.lock").open("w")
    fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = directory / "warm_start_study.json"
    state = read(path) or {"created": now(), "stages": []}
    # Refuse to duplicate an orphan child from an interrupted runner.
    for stage in state["stages"]:
        if stage.get("status") == "running" and stage.get("pid"):
            command_line = Path(f"/proc/{stage['pid']}/cmdline")
            if command_line.exists() and str(directory).encode() in command_line.read_bytes():
                raise RuntimeError(f"Previous child still active: PID {stage['pid']}")
            stage["status"] = "interrupted"
    state.update(status="기존 slow-fast 학습·평가 완료 대기", cumulative_kd_epochs=6,
                 added_epochs=3, teacher="official EMA; unchanged", optimizer="fresh AdamW and schedule")

    def update():
        atomic_json(path, state)
        write_report(directory)

    def execute(name, command, log_path):
        stage = {"name": name, "started": now(), "command": command, "status": "running"}
        state["stages"].append(stage)
        state["status"] = name
        log_path.parent.mkdir(parents=True, exist_ok=True)
        update()
        with log_path.open("a") as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            stage["pid"] = child.pid
            update()
            while child.poll() is None:
                write_report(directory)
                time.sleep(30)
        stage.update(finished=now(), exit_code=child.returncode,
                     status="complete" if child.returncode == 0 else "failed")
        update()
        if child.returncode:
            raise RuntimeError(f"{name} failed; see {log_path}")

    try:
        update()
        original_lock = (directory / "runner.lock").open("a")
        while True:
            try:
                fcntl.flock(original_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                write_report(directory)
                time.sleep(30)
        # A lock release on failure is not permission to skip the original experiment.
        for needed in ("global_cls/train/complete.json", "global_cls/eval/summary.json",
                       "evo_schedule/train/complete.json", "evo_schedule/eval/summary.json"):
            if not read(directory / needed):
                raise RuntimeError(f"Original experiment incomplete: {needed}; resume its runner first")
        weights = directory / "global_cls/train/student.pth"
        state["initial_weights"] = str(weights)
        state["initial_weights_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
        expected = read(directory / "global_cls/eval/summary.json")["weights_sha256"]
        if state["initial_weights_sha256"] != expected:
            raise RuntimeError("Global CLS checkpoint differs from its evaluated weights")
        destination = directory / "global_to_evo"
        python = sys.executable
        if not read(destination / "train/complete.json"):
            command = [python, "-u", "baseline/finetune_drop.py", "--token-method", "evo_schedule",
                "--init-weights", str(weights), "--epochs", "3", "--lr", "5e-5", "--workers", "4",
                "--output-dir", str(destination / "train")]
            if (destination / "train/resume.pth").exists():
                command += ["--resume", str(destination / "train/resume.pth")]
            execute("global_to_evo: KD train", command, destination / "train.log")
        if not read(destination / "eval/summary.json"):
            if (destination / "eval").exists():
                (destination / "eval").rename(destination / f"eval-incomplete-{time.time_ns()}")
            execute("global_to_evo: validation", [python, "-u", "baseline/evaluate_evolution.py",
                "--weights", str(destination / "train/student.pth"), "--methods", "evo_layer", "evo_stage",
                "--ratios", "0.25", "0.5", "0.75", "--latency-samples", "1000",
                "--output-dir", str(destination / "eval")], destination / "evaluate.log")
        state.update(status="이어받기 3에폭 학습·평가 완료", finished=now())
    except Exception as error:
        state["status"] = f"실패: {error}"
        raise
    finally:
        update()


if __name__ == "__main__":
    main()
