#!/usr/bin/env python3
"""Batch-1 latency of each ThinkingViT stage on GPU and CPU, including drop Top-K 6H.

Stages: 3H as in the official early exit (features, head, entropy), 3H with the CLS
attention capture that drop needs, full 6H, and drop Top-K 6H (mask, gather, shorter
sequence). Batch-1 time does not depend on image content, so one random input is timed
repeatedly. float32 with TF32 allowed, like the official latency mode. Median of the runs.

--mode pipeline times the whole batch-1 inference of each method on the same real validation
images (seed-42 order, preprocessed beforehand): 3H, the entropy gate at threshold 1.0, and the
method's 6H pass for routed images. H1 freeze and MLP-selective are the dense-mask oracles
from the H1 branches: they compute every token and discard updates, so they save no time.
"""
import argparse
from datetime import datetime, timezone
import json
import platform
import subprocess
import time

import torch

from evaluate_drop import ROOT, STAGES, NARROW_P, add_paths, load_model, wide_logits
from evaluate_h1_diagnosis import wide_input
from selective_refinement import forward_narrow_with_attention, forward_selective_wide, make_update_mask
from mlp_selective_refinement import forward_mlp_selective_wide

RATIOS = [0.25, 0.5, 0.75]
PIPELINES = {"3H only": None, "Official, threshold 1.0": "full", "6H for every image": "always",
             "H1 freeze Top-K 50%": ("freeze", 0.5), "MLP-selective Top-K 75%": ("mlp", 0.75),
             "Drop Top-K 25%": ("drop", 0.25), "Drop Top-K 50%": ("drop", 0.5), "Drop Top-K 75%": ("drop", 0.75)}
SETTINGS = [("GPU", "cuda", None, 50, 500), ("CPU 4 threads", "cpu", 4, 5, 50), ("CPU 1 thread", "cpu", 1, 3, 30)]


def timed(fn, device, warmup, runs):
    sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - start) * 1000)
    ms = torch.tensor(times, dtype=torch.float64)
    return {"median_ms": ms.median().item(), "p90_ms": ms.quantile(0.9).item(), "runs": runs}


def pipeline(model, x, method):
    """One image end to end; returns whether the 6H pass ran."""
    if method in (None, "full", "always"):
        model.p = NARROW_P
        features = model.forward_features(x, NARROW_P)
        routed = model.entropy(model.forward_head(features)).item() >= 1.0  # gate sync, as in the official exit
        if method == "always" or (method == "full" and routed):
            wide_logits(model, wide_input(model, x, features))
            return True
        return False
    features, logits, attention = forward_narrow_with_attention(model, x, STAGES)
    if model.entropy(logits).item() < 1.0:
        return False
    mode, ratio = method
    mask = make_update_mask(attention, ratio, "topk")
    if mode == "freeze":
        model.forward_head(forward_selective_wide(model, x, features, mask, STAGES))
    elif mode == "mlp":
        model.forward_head(forward_mlp_selective_wide(model, x, features, mask, STAGES))
    else:
        wide_logits(model, wide_input(model, x, features), mask)
    return True


def measure_pipelines(args, images):
    rows = []
    for label, device, threads, _, _ in SETTINGS:
        if threads:
            torch.set_num_threads(threads)
        model = load_model(args.repo.resolve(), args.checkpoint.resolve(), device=device)
        sync = torch.cuda.synchronize if device == "cuda" else (lambda: None)
        batch = [x.to(device) for x in images]
        for name, method in PIPELINES.items():
            for x in batch[:10]:  # warm-up
                pipeline(model, x, method)
            times, routed = [], 0
            for x in batch:
                sync()
                start = time.perf_counter()
                routed += pipeline(model, x, method)
                sync()
                times.append((time.perf_counter() - start) * 1000)
            ms = torch.tensor(times, dtype=torch.float64)
            rows.append({"setting": label, "method": name, "mean_ms": ms.mean().item(), "p50_ms": ms.median().item(),
                         "images": len(batch), "routed_share": routed / len(batch)})
            print(json.dumps(rows[-1]), flush=True)
        del model, batch
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    parser.add_argument("--mode", choices=["stages", "pipeline"], default="stages")
    parser.add_argument("--images", type=int, default=500, help="Validation images per method (pipeline mode)")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    rows = []
    with torch.no_grad():
        if args.mode == "pipeline":
            model = load_model(args.repo.resolve(), args.checkpoint.resolve(), device="cpu")  # puts timm on the path
            from torchvision.datasets import ImageFolder
            from timm.data import create_transform, resolve_data_config
            config = resolve_data_config({}, model=model, use_test_size=True)
            dataset = ImageFolder(args.data_dir / "validation", transform=create_transform(**config, is_training=False))
            order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))[:args.images].tolist()
            rows = measure_pipelines(args, [dataset[i][0].unsqueeze(0) for i in order])
        for label, device, threads, warmup, runs in SETTINGS if args.mode == "stages" else []:
            if threads:
                torch.set_num_threads(threads)
            model = load_model(args.repo.resolve(), args.checkpoint.resolve(), device=device)
            x = torch.randn(1, 3, 224, 224, generator=torch.Generator().manual_seed(0)).to(device)
            features, _, attention = forward_narrow_with_attention(model, x, STAGES)

            def narrow():
                model.p = NARROW_P
                return model.entropy(model.forward_head(model.forward_features(x, NARROW_P)))

            def narrow_attention():
                _, logits, captured = forward_narrow_with_attention(model, x, STAGES)
                return model.entropy(logits), captured

            stages = [("3H", narrow), ("3H + attention", narrow_attention),
                      ("6H full", lambda: wide_logits(model, wide_input(model, x, features)))]
            stages += [(f"6H drop Top-K {round(100 * r)}%",
                        lambda r=r: wide_logits(model, wide_input(model, x, features),
                                                make_update_mask(attention, r, "topk"))) for r in RATIOS]
            for name, fn in stages:
                rows.append({"setting": label, "stage": name, **timed(fn, device, warmup, runs)})
                print(json.dumps(rows[-1]), flush=True)
            del model
    suffix = "-latency" if args.mode == "stages" else "-latency-pipeline"
    output = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + suffix)
    output.mkdir(parents=True)
    cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    environment = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "platform": platform.platform(),
                   "cpu": next((l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.startswith("Model name")), ""),
                   "precision": "float32, TF32 allowed", "batch_size": 1,
                   "measurement": "synchronized wall time of one stage, median; excludes preprocessing and H2D"
                   if args.mode == "stages" else "mean synchronized wall time per image, whole inference including "
                   "the entropy gate; excludes preprocessing and H2D"}
    (output / "summary.json").write_text(json.dumps({"environment": environment, "rows": rows}, indent=2) + "\n")
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
