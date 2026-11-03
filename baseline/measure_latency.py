#!/usr/bin/env python3
"""Batch-1 latency of each ThinkingViT stage on GPU and CPU, including drop Top-K 6H.

Stages: 3H as in the official early exit (features, head, entropy), 3H with the CLS
attention capture that drop needs, full 6H, and drop Top-K 6H (mask, gather, shorter
sequence). Batch-1 time does not depend on image content, so one random input is timed
repeatedly. float32 with TF32 allowed, like the official latency mode. Median of the runs.
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
from selective_refinement import forward_narrow_with_attention, make_update_mask

RATIOS = [0.25, 0.5, 0.75]
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    rows = []
    with torch.no_grad():
        for label, device, threads, warmup, runs in SETTINGS:
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
    output = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-latency")
    output.mkdir(parents=True)
    cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    environment = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "platform": platform.platform(),
                   "cpu": next((l.split(":", 1)[1].strip() for l in cpu.splitlines() if l.startswith("Model name")), ""),
                   "precision": "float32, TF32 allowed", "batch_size": 1,
                   "measurement": "synchronized wall time of one stage, median; excludes preprocessing and H2D"}
    (output / "summary.json").write_text(json.dumps({"environment": environment, "rows": rows}, indent=2) + "\n")
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
