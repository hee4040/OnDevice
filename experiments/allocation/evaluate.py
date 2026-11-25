"""Compare gate-only, ratio-only and joint allocation with the supplied KD checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from efficient_vit.__main__ import atomic, dataset_for
from efficient_vit.model import ROOT, load_native, specialize
from efficient_vit.policy import macs


def prefix(values):
    return np.r_[0, np.cumsum(values, dtype=np.int64)]


def search(entropy, correct, multipliers):
    order = np.argsort(entropy)
    entropy, correct = entropy[order], correct[order]
    n = len(entropy)
    narrow, wide25 = macs(.25)
    _, wide50 = macs(.5)
    exit_ok, low_ok, high_ok = [prefix(column) for column in correct.T]
    baseline_exit = np.searchsorted(entropy, 1.)
    baseline_cost = narrow * n + wide50 * (n - baseline_exit)
    grid = np.unique(np.r_[np.arange(0, 2.31, .025).round(3), 2.5])
    gates, boundaries = np.meshgrid(grid, grid, indexing="ij")
    start = np.searchsorted(entropy, gates)
    stop = np.searchsorted(entropy, boundaries)
    cost = narrow * n + (stop - start) * wide25 + (n - stop) * wide50
    hits = exit_ok[start] + low_ok[stop] - low_ok[start] + high_ok[n] - high_ok[stop]
    masks = {
        "gate_only": gates == boundaries,
        "ratio_only": (gates == 1.) & (boundaries > gates),
        "joint": boundaries > gates,
    }
    results = []
    for multiplier in multipliers:
        limit = baseline_cost * multiplier
        # Match the budget from below. Within 0.25%, select on calibration accuracy.
        policies = {"fixed": (1., 1.)}
        for name, mask in masks.items():
            eligible = mask & (cost <= limit)
            if not eligible.any():
                continue
            closest = cost[eligible].max()
            matched = eligible & (cost >= closest - limit * .0025)
            rows, cols = np.nonzero(matched)
            best = np.lexsort((cost[rows, cols], -hits[rows, cols]))[0]
            policies[name] = (float(gates[rows[best], cols[best]]), float(boundaries[rows[best], cols[best]]))
        results.append({"budget_gmacs": limit / n / 1e9, "multiplier": multiplier, "policies": policies})
    return results


def measure(entropy, correct, gate, boundary):
    codes = (entropy >= gate).astype(np.int64) + (entropy >= boundary)
    counts = np.bincount(codes, minlength=3)
    narrow, wide25 = macs(.25)
    _, wide50 = macs(.5)
    return {
        "top1": 100 * correct[np.arange(len(codes)), codes].mean(),
        "gmacs": (narrow + (counts[1] * wide25 + counts[2] * wide50) / len(codes)) / 1e9,
        "counts_3h_25_50": counts.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "checkpoints/kd_student.pth")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/thinkingvit_3h6h_ema.pth")
    parser.add_argument("--data", type=Path, default=ROOT / "data/imagenet1k")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new file")
    args.samples = 50000
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    native = load_native(args.checkpoint, args.weights)
    model = specialize(native, "cuda", torch.float32)
    dataset, indices = dataset_for(native, args)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    entropies, outcomes = [], []
    with torch.inference_mode():
        for batch, (images, targets) in enumerate(loader, 1):
            images, targets = images.cuda(), targets.cuda()
            features, logits, attention, entropy, embedding = model.narrow(images, return_embedding=True)
            predictions = [logits.argmax(1)]
            for ratio in (.25, .5):
                predictions.append(model.refine(embedding, features, attention, ratio).argmax(1))
            entropies.append(entropy.cpu())
            outcomes.append((torch.stack(predictions, dim=1) == targets[:, None]).cpu())
            if batch % 100 == 0:
                print(f"{min(batch * args.batch_size, len(dataset))}/{len(dataset)}", flush=True)
    entropy = torch.cat(entropies).numpy().astype(np.float64)
    correct = torch.cat(outcomes).numpy()
    midpoint = len(entropy) // 2
    budgets = search(entropy[:midpoint], correct[:midpoint], (.95, 1., 1.05))
    for budget in budgets:
        budget["results"] = {
            name: {
                "gate": gate, "boundary": boundary,
                "calibration": measure(entropy[:midpoint], correct[:midpoint], gate, boundary),
                "heldout": measure(entropy[midpoint:], correct[midpoint:], gate, boundary),
                "full_validation": measure(entropy, correct, gate, boundary),
            }
            for name, (gate, boundary) in budget.pop("policies").items()
        }
    atomic(args.output, {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "split": "first 25k calibration, last 25k held out; seed 42 order",
        "locked_balanced_full_validation": measure(entropy, correct, .8, 1.25),
        "budgets": budgets,
    })
    np.savez_compressed(args.output.with_suffix(".npz"), indices=indices, entropy=entropy, correct=correct)
    print(json.dumps(budgets, indent=2))


if __name__ == "__main__":
    main()
