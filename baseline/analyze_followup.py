#!/usr/bin/env python3
"""Follow-up to the H1 diagnosis: real latency, how much of the drop gain survives
honest checks, and what fine-tuning the drop path adds. Figures + key-number JSON.

Gain = drop accuracy at threshold 1.0 minus the official model's threshold curve at the
same analytic GMACs. Bootstrap: images resampled with the operating point fixed.
Split-half: choose the best drop ratio on one random half, report its gain on the other.
Both evaluate_drop.py runs use the same image order, so per-image results pair up.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from plot_h1_diagnosis import (AQUA, BLUE, MUTED, NARROW_MACS, ORANGE, SURFACE, TEXT, WIDE_FULL_MACS,
                               style, tau_curve, wide_macs)

RATIOS = [10, 25, 50, 75, 90]
SETTINGS = ["GPU", "CPU 4 threads", "CPU 1 thread"]
NOTE = "RTX 4090 / Core i5-13400 · threshold 1.0 · 50,000 ImageNet-1K validation images"


def drop_macs(percent):
    return wide_macs({"ratio": percent / 100, "mode": "drop", "k": 0})


def operating_point(data, percent, routed):
    """Per-image correctness and average GMACs of 3H + drop Top-K 6H for the routed images."""
    key = "wide_correct" if percent == 100 else f"drop_{percent}_correct"
    correct = torch.where(routed, data[key], data["narrow_correct"])
    macs = WIDE_FULL_MACS if percent == 100 else drop_macs(percent)
    return correct, (NARROW_MACS + routed.double().mean().item() * macs) / 1e9


def curve_correct(data, gmacs):
    """Per-image correctness of the threshold curve at the given average GMACs."""
    curve_gmacs, _ = tau_curve(data)
    count = int(torch.argmin((curve_gmacs - gmacs).abs()))
    routed = torch.zeros(len(data["entropy"]), dtype=torch.bool)
    routed[torch.argsort(data["entropy"], descending=True)[:count]] = True
    return torch.where(routed, data["wide_correct"], data["narrow_correct"])


def bootstrap(difference, samples=2000, seed=0):
    generator, means = torch.Generator().manual_seed(seed), []
    for _ in range(samples // 200):
        index = torch.randint(len(difference), (200, len(difference)), generator=generator)
        means.append(difference[index].mean(1))
    means = 100 * torch.cat(means)
    return [means.quantile(0.025).item(), means.quantile(0.975).item()]


def compare(model, reference, percent):
    """Gain over the reference model's threshold curve, with a bootstrap 95% interval."""
    correct, gmacs = operating_point(model, percent, model["entropy"] >= 1.0)
    baseline = curve_correct(reference, gmacs)
    difference = correct.double() - baseline.double()
    top1 = 100 * correct.double().mean().item()
    curve_gmacs, curve_top1 = tau_curve(reference)
    reach = (curve_top1 >= top1).nonzero()  # cheapest threshold that is at least as accurate
    return {"ratio": percent, "gmacs": gmacs, "top1": top1,
            "curve_gmacs_same_top1": curve_gmacs[reach[0, 0]].item() if len(reach) else None,
            "curve_top1": 100 * baseline.double().mean().item(), "gain": 100 * difference.mean().item(),
            "gain_ci95": bootstrap(difference)}


def split_half(data, splits=200, seed=0):
    generator, chosen, held_out = torch.Generator().manual_seed(seed), [], []
    for _ in range(splits):
        order = torch.randperm(len(data["entropy"]), generator=generator)
        halves = [{k: v[part] for k, v in data.items()} for part in order.chunk(2)]
        gains = [[compare_fast(half, p) for p in RATIOS] for half in halves]
        best = max(range(len(RATIOS)), key=lambda i: gains[0][i])
        chosen.append(RATIOS[best])
        held_out.append(gains[1][best])
    held_out = torch.tensor(held_out)
    return {"splits": splits, "held_out_gain_mean": held_out.mean().item(),
            "held_out_gain_p5_p95": [held_out.quantile(0.05).item(), held_out.quantile(0.95).item()],
            "held_out_gain_positive_share": (held_out > 0).double().mean().item(),
            "chosen_ratio_counts": {p: chosen.count(p) for p in RATIOS}}


def compare_fast(data, percent):
    correct, gmacs = operating_point(data, percent, data["entropy"] >= 1.0)
    return 100 * (correct.double().mean() - curve_correct(data, gmacs).double().mean()).item()


def plot_pareto(official, finetuned, key, output):
    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=SURFACE)
    style(ax)
    gmacs, top1 = tau_curve(official)
    ax.plot(gmacs, top1, color=TEXT, linewidth=1.5, label="Threshold sweep (official model)")
    for rows, color, label in ((key["no_retraining"], ORANGE, "Drop Top-K, no retraining"),
                               (key["fine_tuned"], BLUE, "Drop Top-K, fine-tuned")):
        rows = [r for r in rows if r["ratio"] >= 25]  # 10% falls far below the plotted range
        x, y = [r["gmacs"] for r in rows], [r["top1"] for r in rows]
        ax.plot(x, y, color=color, linewidth=1, alpha=0.5)
        ax.scatter(x, y, s=38, color=color, edgecolors=SURFACE, linewidths=1, label=label, zorder=3)
        for r in rows:
            if r["ratio"] in (25, 50, 75):
                ax.annotate(f"{r['ratio']}%", (r["gmacs"], r["top1"]), xytext=(5, -11 if color == ORANGE else 5),
                            textcoords="offset points", color=MUTED, fontsize=8)
    point = key["official_threshold_1"]
    ax.scatter([point["gmacs"]], [point["top1"]], s=46, marker="s", color=SURFACE, edgecolors=TEXT,
               linewidths=1.2, zorder=3, label="Official, threshold 1.0")
    ax.set_xlim(1.55, 3.05)
    ax.set_ylim(70.5, 81.2)
    ax.set_xlabel("Average GMACs per image (analytic, ideal sparse kernels)", color=MUTED, fontsize=9)
    ax.set_ylabel("ImageNet-1K val Top-1 (%)", color=MUTED, fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=TEXT)
    ax.set_title("Fine-tuning the drop path: accuracy per compute", color=TEXT, fontsize=11, loc="left")
    fig.text(0.01, -0.02, NOTE + " · ratio = refined patch tokens", color=MUTED, fontsize=8)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_latency(latency, output):
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    style(ax)
    labels = ["Drop 25%", "Drop 50%", "Drop 75%", "Full 6H"]
    ideal = [100 * drop_macs(p) / WIDE_FULL_MACS for p in (25, 50, 75)] + [100]
    series = [("Ideal (GMACs)", MUTED, ideal)] + [
        (setting, color, [latency[setting][name] / latency[setting]["6H full"] * 100
                          for name in ("6H drop Top-K 25%", "6H drop Top-K 50%", "6H drop Top-K 75%", "6H full")])
        for setting, color in zip(SETTINGS, (BLUE, ORANGE, AQUA))]
    width = 0.2
    for i, (name, color, values) in enumerate(series):
        x = [j + (i - 1.5) * width for j in range(len(labels))]
        ax.bar(x, values, width - 0.03, color=color, label=name)
        for xi, v in zip(x, values):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", color=MUTED, fontsize=7)
    ax.set_xticks(range(len(labels)), labels, color=TEXT)
    ax.set_ylabel("6H stage time (% of full 6H)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 118)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=TEXT, ncol=4)
    ax.set_title("Does dropping tokens make the 6H stage faster? (batch 1)", color=TEXT, fontsize=11, loc="left")
    fig.text(0.01, -0.03, "Median of repeated runs; float32. Includes token selection and gather for drop.",
             color=MUTED, fontsize=8)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True, help="evaluate_drop.py run, official weights")
    parser.add_argument("--finetuned", type=Path, required=True, help="evaluate_drop.py run, --weights")
    parser.add_argument("--finetune-run", type=Path, required=True, help="finetune_drop.py run")
    parser.add_argument("--latency", type=Path, required=True, help="measure_latency.py run")
    parser.add_argument("--official-latency", type=Path, required=True, help="evaluate.py --mode latency run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    official = torch.load(args.official / "per_image.pt", weights_only=True)
    finetuned = torch.load(args.finetuned / "per_image.pt", weights_only=True)
    hard = official["entropy"] >= 1.0  # the official model's routed images, same for both models

    correct, gmacs = operating_point(official, 100, hard)
    key = {"official_threshold_1": {"gmacs": gmacs, "top1": 100 * correct.double().mean().item()},
           "no_retraining": [compare(official, official, p) for p in RATIOS],
           "fine_tuned": [compare(finetuned, official, p) for p in RATIOS],
           "hard_images": {"count": int(hard.sum()), **{
               f"{name}_{label}": 100 * data[k][hard].double().mean().item()
               for name, data in (("official", official), ("fine_tuned", finetuned))
               for label, k in [("3h", "narrow_correct"), ("full_6h", "wide_correct")] +
               [(f"drop_{p}", f"drop_{p}_correct") for p in RATIOS]}},
           "all_images": {f"{name}_{label}": 100 * data[k].double().mean().item()
                          for name, data in (("official", official), ("fine_tuned", finetuned))
                          for label, k in (("3h", "narrow_correct"), ("full_6h", "wide_correct"))},
           "routed_share": {"official": hard.double().mean().item(),
                            "fine_tuned": (finetuned["entropy"] >= 1.0).double().mean().item()},
           "split_half_no_retraining": split_half(official)}
    rows = json.loads((args.latency / "summary.json").read_text())
    latency = {s: {r["stage"]: r["median_ms"] for r in rows["rows"] if r["setting"] == s} for s in SETTINGS}
    share = key["routed_share"]["official"]
    key["latency"] = {"environment": rows["environment"], "stage_median_ms": latency, "pipeline_threshold_1_ms": {
        s: {"official": t["3H"] + share * t["6H full"],
            **{f"drop_{p}": t["3H + attention"] + share * t[f"6H drop Top-K {p}%"] for p in (25, 50, 75)}}
        for s, t in latency.items()}}
    key["latency"]["official_evaluate_py"] = json.loads((args.official_latency / "summary.json").read_text())
    key["finetune"] = {**json.loads((args.finetune_run / "arguments.json").read_text()),
                       "last_log": json.loads((args.finetune_run / "train_log.json").read_text())[-1]}
    (args.output / "followup_key_numbers.json").write_text(json.dumps(key, indent=2) + "\n")
    plot_pareto(official, finetuned, key, args.output / "followup_pareto.png")
    plot_latency(latency, args.output / "followup_latency.png")
    print(json.dumps({k: key[k] for k in ("no_retraining", "fine_tuned", "hard_images", "all_images",
                                           "routed_share", "split_half_no_retraining")}, indent=1))
    print(json.dumps(key["latency"]["pipeline_threshold_1_ms"], indent=1))


if __name__ == "__main__":
    main()
