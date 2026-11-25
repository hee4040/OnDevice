#!/usr/bin/env python3
"""Evaluate Evo-ViT adaptations with the established seed-42 ImageNet ordering."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from evaluate_drop import ROOT, STAGES, add_paths, load_model, wide_logits
from evaluate_h1_diagnosis import wide_input
from selective_refinement import forward_narrow_with_attention
from token_evolution import METHODS, evolution_logits


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                             "--format=csv,noheader"], capture_output=True, text=True)
    return [line for line in result.stdout.splitlines() if line.split(",")[0].strip() != str(os.getpid())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=["last_cls", "global_cls", "evo_layer", "evo_stage"])
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cls-decay", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--latency-samples", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.samples <= 50000 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid evaluation size")
    if any(not 0 < r <= 1 for r in args.ratios) or not 0 <= args.cls_decay <= 1:
        parser.error("Invalid ratios or attention decay")
    if args.output_dir.exists():
        parser.error("Output exists; preserve it and choose a new output directory")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model = load_model(args.repo.resolve(), args.checkpoint.resolve(), args.weights)
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    from timm.data import create_transform, resolve_data_config
    dataset = ImageFolder(args.data_dir / "validation", transform=create_transform(
        **resolve_data_config({}, model=model, use_test_size=True), is_training=False))
    if len(dataset) != 50000:
        raise RuntimeError("Expected complete validation split")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:args.samples].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    args.output_dir.mkdir(parents=True)
    values = {"entropy": [], "narrow_correct": [], "wide_correct": []}
    for method in args.methods:
        for ratio in args.ratios:
            values[f"{method}_{round(ratio * 100)}_correct"] = []
    started = time.time()
    with torch.inference_mode():
        for batch, (images, labels) in enumerate(loader, 1):
            images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            features, narrow, last = forward_narrow_with_attention(model, images)
            global_features, global_narrow, scores = forward_narrow_with_attention(
                model, images, global_cls=True, decay=args.cls_decay)
            if batch == 1:
                torch.testing.assert_close(features, global_features, rtol=0, atol=0)
                torch.testing.assert_close(narrow, global_narrow, rtol=0, atol=0)
            x0 = wide_input(model, images, features)
            full = wide_logits(model, x0)
            values["entropy"].append(model.entropy(narrow)[:, 0].cpu())
            values["narrow_correct"].append((narrow.argmax(1) == labels).cpu())
            values["wide_correct"].append((full.argmax(1) == labels).cpu())
            for method in args.methods:
                for ratio in args.ratios:
                    output = evolution_logits(model, x0, last if method == "last_cls" else scores,
                                              ratio, method, decay=args.cls_decay)
                    values[f"{method}_{round(ratio * 100)}_correct"].append((output.argmax(1) == labels).cpu())
            if batch % 100 == 0:
                print(f"Evaluated {min(batch * args.batch_size, args.samples)}/{args.samples}", flush=True)
    values = {key: torch.cat(parts) for key, parts in values.items()}
    torch.save({**values, "indices": torch.tensor(indices)}, args.output_dir / "per_image.pt")
    routed = values["entropy"] >= 1
    summary = {"samples": args.samples, "seed": 42, "threshold": 1.0, "methods": args.methods,
               "weights": str(args.weights), "weights_sha256": digest(args.weights),
               "checkpoint_sha256": digest(args.checkpoint), "gpu": torch.cuda.get_device_name(),
               "torch": torch.__version__, "precision": "FP32, TF32 allowed", "cls_decay": args.cls_decay,
               "routed": int(routed.sum()), "elapsed_seconds": time.time() - started,
               "finished": datetime.now(timezone.utc).isoformat(), "top1": {}}
    for key, correct in values.items():
        if not key.endswith("_correct"):
            continue
        summary["top1"][key] = {"all": 100 * correct.double().mean().item(),
            "threshold_1": 100 * torch.where(routed, correct, values["narrow_correct"]).double().mean().item()}
    if args.latency_samples:
        processes_before = gpu_processes()
        timing_loader = DataLoader(Subset(dataset, indices[:args.latency_samples + 50]), batch_size=1,
                                   num_workers=args.workers, pin_memory=True)
        timings = {m: [] for m in ["full"] + args.methods}
        with torch.inference_mode():
            for index, (images, _) in enumerate(timing_loader):
                images = images.cuda()
                order = list(timings)
                order = order[index % len(order):] + order[:index % len(order)]
                for method in order:
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    if method == "full":
                        model.p = 0.5
                        features = model.forward_features(images, 0.5)
                        narrow = model.forward_head(features)
                    else:
                        features, narrow, score = forward_narrow_with_attention(
                            model, images, global_cls=method != "last_cls", decay=args.cls_decay)
                    if model.entropy(narrow).item() >= 1:
                        x0 = wide_input(model, images, features)
                        if method == "full":
                            output = wide_logits(model, x0)
                        else:
                            output = evolution_logits(model, x0, score, 0.5, method, decay=args.cls_decay)
                    torch.cuda.synchronize()
                    if index >= 50:
                        timings[method].append((time.perf_counter() - t) * 1000)
        summary["latency"] = {"samples": len(next(iter(timings.values()))), "warmup": 50,
            "batch_size": 1, "ratio": 0.5, "threshold": 1, "preprocessing_h2d_excluded": True,
            "mean_ms": {key: sum(v) / len(v) for key, v in timings.items()},
            "other_gpu_processes_before": processes_before, "other_gpu_processes_after": gpu_processes()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
