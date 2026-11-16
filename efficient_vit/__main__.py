import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import threading
import time
import torch
from .model import ROOT, load_native, specialize
from .policy import PRESETS, macs
from .runtime import Engine
from .verify import verify, NativeEngine, native_narrow


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def processes():
    rows = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True)
    return [line for line in rows.splitlines() if line.split(",")[0].strip() != str(os.getpid())]


def dataset_for(native, args):
    from torchvision.datasets import ImageFolder
    from torch.utils.data import Subset
    from timm.data import create_transform, resolve_data_config
    dataset = ImageFolder(args.data / "validation", transform=create_transform(
        **resolve_data_config({}, model=native, use_test_size=True), is_training=False))
    if len(dataset) != 50000:
        raise ValueError("Expected ImageNet validation 50,000 images")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:args.samples].tolist()
    return Subset(dataset, indices), indices


def main():
    parser = argparse.ArgumentParser(description="Efficient KD ThinkingViT: evaluate, validate, or benchmark")
    parser.add_argument("command", choices=("validate", "evaluate", "benchmark"))
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/kd_student.pth")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/thinkingvit_3h6h_ema.pth")
    parser.add_argument("--official", action="store_true", help="Use official EMA instead of KD weights")
    parser.add_argument("--data", type=Path, default=ROOT / "data/imagenet1k")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--runtime", choices=("eager", "graph", "native"), default="graph")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--policies", nargs="+", choices=tuple(PRESETS), default=list(PRESETS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--real-images", action="store_true", help="Validate native/eager/Graph on 32 real images as well")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.runtime == "native" and args.command == "validate":
        parser.error("Validate eager/Graph against native, rather than native against itself")
    args.samples = args.samples or (50000 if args.command == "evaluate" else 1000)
    if not 1 <= args.samples <= 50000 or args.batch_size < 1 or args.workers < 0 or args.repeats < 1:
        parser.error("Invalid samples, batch size, workers or repeats")
    args.output = args.output or ROOT / "results/efficient-kd" / f"{args.command}-{time.time_ns()}.json"
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.distributions.Distribution.set_default_validate_args(False)
    if args.command == "benchmark" and processes():
        raise RuntimeError("Other GPU compute processes active; benchmark later")
    dtype = torch.float32 if args.precision == "fp32" else torch.float16
    native = load_native(args.checkpoint, None if args.official else args.weights)
    model = specialize(native, "cuda", dtype)
    native = native.cuda().to(dtype)
    result = {"started": datetime.now(timezone.utc).isoformat(), "command": args.command,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "weights_sha256": None if args.official else hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "precision": args.precision,
        "policies": {name: asdict(PRESETS[name]) for name in args.policies},
        "mac_convention": "1 MAC = multiply+accumulate; 384-channel embedding cached after narrow (native control recomputes); excludes norm/softmax/sort/gather/memory traffic",
        "policy_selection": "locked source policies, no tuning on this evaluation"}
    if args.command == "validate":
        engine = Engine(model, args.runtime)
        x = torch.randn(1, 3, 224, 224, device="cuda", dtype=dtype)
        result["max_absolute_errors"] = verify(native, model, x, engine if args.runtime != "eager" else None)
        if args.real_images:
            dataset, _ = dataset_for(native, args)
            maximum = dict(result["max_absolute_errors"])
            for index in range(min(32, len(dataset))):
                x = dataset[index][0].unsqueeze(0).cuda().to(dtype)
                values = verify(native, model, x, engine if args.runtime != "eager" else None)
                for key, value in values.items():
                    maximum[key] = max(maximum.get(key, 0), value)
            result["max_absolute_errors"] = maximum
            result["real_images_checked"] = min(32, len(dataset))
        result["status"] = "passed"
    else:
        dataset, indices = dataset_for(native, args)
        result["samples"], result["indices"] = len(dataset), indices
        if args.command == "evaluate":
            from torch.utils.data import DataLoader
            engine = NativeEngine(native) if args.runtime == "native" else Engine(model, "eager")
            result["runtime"] = "native" if args.runtime == "native" else "eager"
            loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
            results = {name: {"correct": [], "codes": [], "predicted": []} for name in args.policies}
            with torch.inference_mode():
                for batch, (images, targets) in enumerate(loader, 1):
                    x, y = images.cuda().to(dtype), targets.cuda()
                    narrow = native_narrow(native, x) if args.runtime == "native" else model.narrow(x, return_embedding=True)
                    for name in args.policies:
                        logits, codes, entropy = engine.predict(x, name, narrow)
                        results[name]["correct"].append((logits.argmax(1) == y).cpu())
                        results[name]["codes"].append(codes.cpu())
                        results[name]["predicted"].append(logits.argmax(1).cpu())
                    if batch % 100 == 0:
                        print(f"{min(batch * args.batch_size, len(dataset))}/{len(dataset)}", flush=True)
            results = {name: {key: torch.cat(parts) for key, parts in values.items()} for name, values in results.items()}
            torch.save({"indices": torch.tensor(indices), "policies": results}, args.output.with_suffix(".pt"))
            result["accuracy"] = {}
            for name, values in results.items():
                policy = PRESETS[name]
                cost = torch.full((len(dataset),), float(macs(.5)[0]), dtype=torch.float64)
                counts = {"exit": int((values["codes"] == -1).sum())}
                for code, ratio in enumerate(policy.ratios):
                    selected = values["codes"] == code
                    cost[selected] += macs(ratio, cached_embedding=args.runtime != "native")[1]
                    counts[str(ratio)] = int(selected.sum())
                result["accuracy"][name] = {"top1": 100 * values["correct"].double().mean().item(),
                    "gmacs": cost.mean().item() / 1e9, "path_counts": counts}
        else:
            # Resident model/input memory is excluded from incremental peak working memory.
            # The absolute allocation for this process is also recorded separately.
            if args.runtime != "native":
                del native
            inputs = [dataset[i][0].unsqueeze(0).cuda().to(dtype) for i in range(len(dataset))]
            start = time.perf_counter()
            engine = NativeEngine(native) if args.runtime == "native" else Engine(model, args.runtime)
            reference_engine = Engine(model, "eager")
            with torch.inference_mode():
                for name in args.policies:
                    for x in inputs[:32]:
                        expected = reference_engine.predict(x, name)
                        actual = engine.predict(x, name)
                        torch.testing.assert_close(actual[0], expected[0], rtol=1e-4 if dtype == torch.float32 else .03,
                                                   atol=2e-5 if dtype == torch.float32 else .02)
                        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
            result["runtime_matches_eager_on_real_images"] = 32
            del reference_engine
            if args.runtime == "native":
                del model
            result.update(runtime=args.runtime, setup_seconds=time.perf_counter() - start,
                measurement="batch1 synchronized wall-clock; preprocessing/H2D/final argmax excluded; gate included",
                rows=[], gpu_monitor=[])
            event = threading.Event()
            def monitor():
                while not event.is_set():
                    try:
                        result["gpu_monitor"].append({"time": time.time(), "other_processes": processes()})
                    except Exception as error:
                        result.setdefault("monitor_errors", []).append(str(error))
                    event.wait(1)
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            try:
                with torch.inference_mode():
                    for repeat in range(args.repeats):
                        names = list(args.policies)
                        random.Random(42 + repeat).shuffle(names)
                        for name in names:
                            for x in inputs[:50]:
                                engine.predict(x, name)
                            torch.cuda.synchronize()
                            base = torch.cuda.memory_allocated()
                            torch.cuda.reset_peak_memory_stats()
                            times = []
                            for x in inputs:
                                torch.cuda.synchronize()
                                tick = time.perf_counter()
                                output = engine.predict(x, name)
                                torch.cuda.synchronize()
                                times.append((time.perf_counter() - tick) * 1000)
                            row = {"policy": name, "repeat": repeat, "mean_ms": statistics.mean(times),
                                "p95_ms": torch.tensor(times).quantile(.95).item(),
                                "serial_images_per_s": 1000 / statistics.mean(times),
                                "resident_allocated_mib": base / 2**20,
                                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                                "incremental_peak_mib": max(0, torch.cuda.max_memory_allocated() - base) / 2**20}
                            result["rows"].append(row)
                            atomic(args.output.with_suffix(".partial.json"), result)
                            print(json.dumps(row), flush=True)
            finally:
                event.set()
                thread.join(timeout=3)
            result["clean"] = not result.get("monitor_errors") and not any(s["other_processes"] for s in result["gpu_monitor"])
            result["latency_ms"] = {name: statistics.median(r["mean_ms"] for r in result["rows"] if r["policy"] == name) for name in args.policies}
    result["finished"] = datetime.now(timezone.utc).isoformat()
    atomic(args.output, result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("indices", "gpu_monitor")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
