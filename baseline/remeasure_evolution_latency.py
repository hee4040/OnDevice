#!/usr/bin/env python3
"""Paired latency sweep with continuous GPU-process monitoring and safe training suspension."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import statistics
import subprocess
import threading
import time

import torch
from evaluate_drop import ROOT, load_model, wide_logits
from evaluate_h1_diagnosis import wide_input
from selective_refinement import forward_narrow_with_attention
from token_evolution import evolution_logits


def now():
    return datetime.now(timezone.utc).isoformat()


def snapshot():
    query = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                     "--format=csv,noheader"], text=True)
    processes = []
    for line in query.splitlines():
        pid = int(line.split(",")[0])
        status = Path(f"/proc/{pid}/status")
        state = next((s.split()[1] for s in status.read_text().splitlines() if s.startswith("State:")), "?") if status.exists() else "exited"
        processes.append({"pid": pid, "state": state, "details": line})
    environment = subprocess.check_output(["nvidia-smi",
        "--query-gpu=temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader"], text=True).strip()
    return {"time": now(), "processes": processes, "temperature_utilization_power_clocks": environment}


def infer(model, images, method):
    if method == "full":
        model.p = .5
        features = model.forward_features(images, .5)
        logits = model.forward_head(features)
        scores = None
    else:
        features, logits, scores = forward_narrow_with_attention(model, images, global_cls=method != "last_cls")
    routed = model.entropy(logits).item() >= 1
    if routed:
        x0 = wide_input(model, images, features)
        logits = wide_logits(model, x0) if method == "full" else evolution_logits(model, x0, scores, .5, method)
    return logits, routed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/evo-vit-20261115")
    parser.add_argument("--pause-pid", type=int)
    parser.add_argument("--wait-for-all", action="store_true")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.samples < 1 or args.repeats < 1:
        parser.error("Choose a new output file and positive sample/repeat counts")
    if args.wait_for_all:
        while not (args.directory / "global_to_evo/eval/summary.json").exists():
            time.sleep(30)
        # Let the owning runner finish before starting the final sweep.
        import fcntl
        lock = (args.directory / "warm_start_runner.lock").open("a")
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = {"started": now(), "status": "preparing", "samples": args.samples, "repeats": args.repeats,
                "batch_size": 1, "warmup": 50, "ratio": .5, "threshold": 1,
                "precision": "FP32, TF32 allowed", "cpu_threads": 1, "seed": 42,
                "scope": "preprocessing/H2D/argmax excluded; gate included; synchronized wall-clock",
                "paused_training_pid": args.pause_pid, "monitor": [], "rows": []}
    paused = False
    stop = threading.Event()
    monitor_thread = None

    def save():
        tmp = args.output.with_suffix(".tmp")
        tmp.write_text(json.dumps(evidence, indent=2) + "\n")
        tmp.replace(args.output)

    def monitor():
        while not stop.is_set():
            try:
                evidence["monitor"].append(snapshot())
            except Exception as error:
                evidence.setdefault("monitor_errors", []).append(str(error))
            stop.wait(1)

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    try:
        if args.pause_pid:
            command = Path(f"/proc/{args.pause_pid}/cmdline").read_bytes().replace(b"\0", b" ")
            if b"baseline/finetune_drop.py" not in command or str(ROOT).encode() not in command:
                raise RuntimeError("Pause target is not this project's training process")
            evidence["paused_command"] = command.decode()
            os.kill(args.pause_pid, signal.SIGSTOP)
            paused = True
            time.sleep(3)
        # Drain outstanding training work and allow the laptop GPU to cool.
        deadline = time.monotonic() + 45
        while True:
            sample = snapshot()
            evidence["monitor"].append(sample)
            active = [p for p in sample["processes"] if p["pid"] != os.getpid()
                      and not (p["pid"] == args.pause_pid and p["state"] in ("T", "t"))]
            if active:
                if args.wait_for_all:
                    time.sleep(30)
                    continue
                raise RuntimeError(f"Other GPU processes active: {active}")
            temperature = float(sample["temperature_utilization_power_clocks"].split(",")[0])
            if temperature <= 65 or time.monotonic() >= deadline:
                break
            time.sleep(3)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        checkpoint = ROOT / "checkpoints/thinkingvit_3h6h_ema.pth"
        repo = ROOT / "third_party/ThinkingViT"
        definitions = [ ("official", None, ["full"]),
            ("existing_kd", ROOT / "results/drop-ft-20261106/kd/train/student.pth",
             ["full", "last_cls", "global_cls", "evo_layer", "evo_stage"]) ]
        for name, methods in [("global_cls", ["full", "last_cls", "global_cls"]),
                              ("evo_schedule", ["full", "evo_layer", "evo_stage"]),
                              ("global_to_evo", ["full", "evo_layer", "evo_stage"])]:
            if (args.directory / name / "eval/summary.json").exists():
                definitions.append((name, args.directory / name / "train/student.pth", methods))
        models, cases, provenance = {}, {}, {}
        for name, weights, methods in definitions:
            models[name] = load_model(repo, checkpoint, weights)
            provenance[name] = {"weights": str(weights or checkpoint),
                "sha256": hashlib.sha256((weights or checkpoint).read_bytes()).hexdigest()}
            for method in methods:
                cases[f"{name}/{method}"] = (name, method)
        evidence.update(models=provenance, gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                        cuda=torch.version.cuda, status="measuring")
        from torchvision.datasets import ImageFolder
        from timm.data import create_transform, resolve_data_config
        dataset = ImageFolder(ROOT / "data/imagenet1k/validation", transform=create_transform(
            **resolve_data_config({}, model=models["official"], use_test_size=True), is_training=False))
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:args.samples].tolist()
        evidence["indices"] = indices
        images = [dataset[i][0].unsqueeze(0).cuda() for i in indices]
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
        with torch.inference_mode():
            for repeat in range(args.repeats):
                order = list(cases)
                random.Random(42 + repeat).shuffle(order)
                for case in order:
                    name, method = cases[case]
                    model = models[name]
                    for x in images[:50]:
                        infer(model, x, method)
                    times, routed_count = [], 0
                    for x in images:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        logits, routed = infer(model, x, method)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) * 1000)
                        routed_count += routed
                    values = torch.tensor(times, dtype=torch.float64)
                    row = {"case": case, "repeat": repeat, "mean_ms": statistics.mean(times),
                        "p50_ms": values.median().item(), "p95_ms": values.quantile(.95).item(),
                        "routed": routed_count}
                    evidence["rows"].append(row)
                    save()
                    print(json.dumps(row), flush=True)
        stop.set()
        monitor_thread.join(timeout=3)
        active_samples = [sample for sample in evidence["monitor"] if any(
            p["pid"] != os.getpid() and not (p["pid"] == args.pause_pid and p["state"] in ("T", "t"))
            for p in sample["processes"])]
        evidence["clean"] = not active_samples and not evidence.get("monitor_errors")
        evidence["contaminated_samples"] = active_samples
        evidence["aggregate"] = {}
        for case in cases:
            means = [r["mean_ms"] for r in evidence["rows"] if r["case"] == case]
            evidence["aggregate"][case] = {"median_mean_ms": statistics.median(means),
                                          "min_mean_ms": min(means), "max_mean_ms": max(means)}
        evidence["status"] = "complete" if evidence["clean"] else "contaminated"
    except BaseException as error:
        evidence.update(status="failed", error=str(error))
        raise
    finally:
        stop.set()
        if monitor_thread:
            monitor_thread.join(timeout=3)
        if paused:
            os.kill(args.pause_pid, signal.SIGCONT)
            evidence["training_resumed"] = now()
        evidence["finished"] = now()
        save()
        from evolution_report import write_report
        write_report(args.directory)


if __name__ == "__main__":
    main()
