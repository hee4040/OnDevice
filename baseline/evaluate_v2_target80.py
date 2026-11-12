#!/usr/bin/env python3
"""Independent check: apply the locked target80 policies (chosen on ImageNet-1K val) to ImageNet-V2 matched-frequency
(10,000 never-seen images). No retuning. Same student weights, preprocessing and real conditional execution as evaluate_target80.py."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from evaluate_drop import add_paths, load_model
from selective_refinement import forward_narrow_with_attention
from target80_runtime import forward_target80
from target80_gate_search import COSTS
from analyze_adaptive_drop import NARROW_MACS, save_json, save_csv

REFERENCE = {"official_tau1_full6h": {"kind": "fixed", "gate": 1.0, "ratio": 1.0},   # original ThinkingViT routing, all 196 patches
             "all_images_full6h": {"kind": "fixed", "gate": "-inf", "ratio": 1.0}}  # upper bound: every image to 6H


def main():
    ap = argparse.ArgumentParser(description=__doc__); add_paths(ap)
    ap.add_argument("--weights", type=Path, required=True); ap.add_argument("--policies", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    policies = {**json.loads(args.policies.read_text()), **REFERENCE}
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.repo.resolve(), args.checkpoint.resolve(), args.weights)
    from timm.data import create_transform, resolve_data_config
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    cfg = resolve_data_config({}, model=model, use_test_size=True)
    ds = ImageFolder(args.data_dir / "validation", transform=create_transform(**cfg, is_training=False))
    assert len(ds) == 10000 and ds.classes == [f"{i:04d}" for i in range(1000)], (len(ds), ds.classes[:3])
    loader = DataLoader(ds, batch_size=64, num_workers=4, pin_memory=True)
    saved = {k: {"correct": [], "codes": [], "entropy": []} for k in policies}; narrow_correct = []
    with torch.no_grad():
        for bi, (images, target) in enumerate(loader, 1):
            images = images.cuda(non_blocking=True); narrow = forward_narrow_with_attention(model, images)
            narrow_correct.append(narrow[1].argmax(1).cpu() == target)
            for name, p in policies.items():
                logits, codes, e = forward_target80(model, images, p, narrow)
                saved[name]["correct"].append(logits.argmax(1).cpu() == target); saved[name]["codes"].append(codes.cpu()); saved[name]["entropy"].append(e.cpu())
            if bi % 40 == 0: print(f"V2 {bi * 64}/10000", flush=True)
    saved = {n: {k: torch.cat(v).numpy() for k, v in d.items()} for n, d in saved.items()}
    narrow_correct = torch.cat(narrow_correct).numpy()
    base = saved["baseline"]; base_cost = int(NARROW_MACS) * 10000 + int(COSTS[base["codes"][base["codes"] >= 0]].sum())
    rng = np.random.default_rng(0); boot = rng.integers(0, 10000, size=(5000, 10000))
    rows = [{"method": "3H_only", "correct": int(narrow_correct.sum()), "top1": 100 * narrow_correct.mean(), "gmacs": float(NARROW_MACS) / 1e9}]
    for name, d in saved.items():
        c = d["codes"]; cost = int(NARROW_MACS) * 10000 + int(COSTS[c[c >= 0]].sum()); diff = d["correct"].astype(float) - base["correct"]
        ci = np.percentile(diff[boot].mean(1) * 100, [2.5, 97.5])
        rows.append({"method": name, "correct": int(d["correct"].sum()), "top1": 100 * d["correct"].mean(), "gmacs": cost / 1e4 / 1e9,
                     "cost_change_percent": 100 * (cost / base_cost - 1), "stage2_fraction": float((c >= 0).mean()),
                     "diff_vs_baseline_pp": 100 * diff.mean(), "ci95_pp": [float(ci[0]), float(ci[1])],
                     "improved": int((d["correct"] & ~base["correct"]).sum()), "regressed": int((~d["correct"] & base["correct"]).sum())})
    import pandas as pd; pd.DataFrame(rows).to_csv(args.output / "comparison.csv", index=False)
    save_json(args.output / "summary.json", {"dataset": "ImageNet-V2 matched-frequency (vaishaal/ImageNetV2, folders zero-padded)", "n": 10000,
              "policies": policies, "comparison": rows, "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
              "bootstrap": "paired, 5000 resamples, seed 0, policies fixed (no selection uncertainty)",
              "note": "Policies locked from ImageNet-1K validation search; not retuned on V2."})
    torch.save({"methods": saved, "narrow_correct": narrow_correct, "targets": np.array(ds.targets)}, args.output / "per_image.pt")
    for r in rows: print(r, flush=True)


if __name__ == "__main__":
    main()
