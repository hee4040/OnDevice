#!/usr/bin/env python3
"""Turn an evaluate_h1_diagnosis.py run into figures, a CSV and a key-number JSON.

GMACs are analytic counts for an ideal sparse implementation (patch embedding, blocks,
3H->6H projection and head; norms and softmax ignored). Freeze still computes K/V for
frozen tokens, drop shortens the sequence, MLP-selective runs attention on every token.
The tau curve routes the most uncertain images first, which is every possible threshold.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torch

TOKENS, PATCHES, NARROW, WIDE, DEPTH, CLASSES = 197, 196, 192, 384, 12, 1000
POLICIES = [("topk", "CLS attention Top-K", "#2a78d6"), ("random", "Random-K", "#1baf7a"),
            ("bottomk", "CLS attention Bottom-K", "#eb6834")]
RATIO_COLORS = {0.25: "#86b6ef", 0.5: "#2a78d6", 0.75: "#104281"}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NOTE = "RTX 4090 · threshold 1.0 · 50,000 ImageNet-1K validation images · no retraining"


def block_macs(d, queries, keys, mlp):
    return (2 * queries + 2 * keys + 8 * mlp) * d * d + 2 * queries * keys * d


def stage_macs(d, blocks):
    return PATCHES * 3 * 16 * 16 * d + sum(block_macs(d, *b) for b in blocks) + d * CLASSES


FULL = (TOKENS, TOKENS, TOKENS)
NARROW_MACS = stage_macs(NARROW, [FULL] * DEPTH)
WIDE_FULL_MACS = TOKENS * NARROW * WIDE + stage_macs(WIDE, [FULL] * DEPTH)


def wide_macs(row):
    n = 1 + round(PATCHES * row["ratio"])
    sparse = {"freeze": (n, TOKENS, n), "drop": (n, n, n), "mlp": (TOKENS, TOKENS, n)}[row["mode"]]
    return TOKENS * NARROW * WIDE + stage_macs(WIDE, [FULL] * row["k"] + [sparse] * (DEPTH - row["k"]))


def tau_curve(data):
    narrow, wide = data["narrow_correct"].double(), data["wide_correct"].double()
    gain = (wide - narrow)[torch.argsort(data["entropy"], descending=True)]
    top1 = 100 * (narrow.sum() + torch.cat([torch.zeros(1, dtype=torch.double), gain.cumsum(0)])) / len(gain)
    fraction = torch.arange(len(gain) + 1, dtype=torch.double) / len(gain)
    return (NARROW_MACS + fraction * WIDE_FULL_MACS) / 1e9, top1


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def reference_lines(ax, summary, x_text):
    for value, label in ((summary["threshold_top1"], "all tokens refined (official)"),
                         (summary["narrow_only_top1"], "skip 6H, keep the 3H answer")):
        ax.axhline(value, color=MUTED, linewidth=0.75, linestyle="-" if "official" in label else "--")
        ax.text(x_text, value + 0.4, f"{label} ({value:.2f})", color=MUTED, fontsize=8, va="bottom")


def finish(fig, axes, title, output, legend_axis=-1):
    handles, labels = axes[legend_axis].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=TEXT, fontsize=12, x=0.01, ha="left", y=1.07)
    fig.text(0.01, -0.04, NOTE, color=MUTED, fontsize=8)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def line(ax, x, y, color, label, **kwargs):
    ax.plot(x, y, color=color, linewidth=1.5, marker=kwargs.pop("marker", "o"), markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1, label=label, **kwargs)


def plot_freeze_drop(rows, summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    titles = {"freeze": "Freeze: unselected tokens stay as stale K/V (H1)",
              "drop": "Drop: unselected tokens leave the 6H pass"}
    for ax, mode in zip(axes, ("freeze", "drop")):
        style(ax)
        reference_lines(ax, summary, 1)
        start = rows[f"{mode}_k0_topk_0"]["overall_top1"]
        for policy, label, color in POLICIES:
            points = sorted((100 * r["ratio"], r["overall_top1"]) for r in rows.values()
                            if r["experiment"] == "freeze_drop" and r["mode"] == mode and r["policy"] == policy
                            and r["ratio"] > 0)
            x, y = zip(*([(0, start)] + points + [(100, summary["threshold_top1"])]))
            line(ax, x, y, color, label)
        ax.set_title(titles[mode], color=TEXT, fontsize=10.5, loc="left")
        ax.set_xlabel("Refined patch tokens (% of 196)", color=MUTED, fontsize=9)
    axes[0].set_ylabel("ImageNet-1K val Top-1 (%)", color=MUTED, fontsize=9)
    finish(fig, axes, "Experiment 1: freezing vs dropping the unselected tokens", output)


def plot_full_k(rows, summary, output):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    titles = {"freeze": "All tokens for k blocks, then freeze", "drop": "All tokens for k blocks, then drop"}
    for ax, mode in zip(axes, ("freeze", "drop")):
        style(ax)
        reference_lines(ax, summary, 0.1)
        for ratio, policy, color, style_ in [(r, "topk", c, "-") for r, c in RATIO_COLORS.items()] + \
                                            [(0.5, "random", AQUA, "--")]:
            name = lambda k: f"{mode}_k{k}_{policy}_{round(100 * ratio)}"
            x = [0, 2, 4, 6, 8, 10, 12]
            y = [rows[name(k)]["overall_top1"] for k in x[:-1]] + [summary["threshold_top1"]]
            label = f"Top-K {round(100 * ratio)}%" if policy == "topk" else "Random-K 50%"
            line(ax, x, y, color, label, linestyle=style_)
        ax.set_title(titles[mode], color=TEXT, fontsize=10.5, loc="left")
        ax.set_xlabel("k = 6H blocks that update every token (of 12)", color=MUTED, fontsize=9)
        ax.set_xticks([0, 2, 4, 6, 8, 10, 12])
    axes[0].set_ylabel("ImageNet-1K val Top-1 (%)", color=MUTED, fontsize=9)
    finish(fig, axes, "Experiment 2: how many 6H blocks must see every token", output)


def plot_pareto(rows, summary, curve, output):
    fig, (ax, zoom) = plt.subplots(1, 2, figsize=(12, 5), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.15, 1]})
    style(ax)
    style(zoom)
    gmacs, top1 = curve
    for axis in (ax, zoom):
        axis.plot(gmacs, top1, color=TEXT, linewidth=1.5, label="Threshold sweep (6H for fewer images)")
    series = [
        (lambda r: r["experiment"] == "freeze_drop" and r["mode"] == "freeze" and r["policy"] == "topk",
         BLUE, "o", "H1 freeze, Top-K, ratio 0-99%"),
        (lambda r: r["experiment"] == "freeze_drop" and r["mode"] == "drop" and r["policy"] == "topk",
         ORANGE, "o", "Drop, Top-K, ratio 0-99%"),
        (lambda r: r["experiment"] == "full_k" and r["mode"] == "drop" and r["policy"] == "topk",
         ORANGE, "s", "Drop after k full blocks, Top-K"),
        (lambda r: r["mode"] == "mlp" and r["policy"] == "topk", AQUA, "o", "MLP-selective, Top-K"),
    ]
    for keep, color, marker, label in series:
        points = [(r["avg_gmacs"], r["overall_top1"]) for r in rows.values() if keep(r)]
        x, y = zip(*points)
        for axis in (ax, zoom):
            axis.scatter(x, y, s=34, color=color if marker == "o" else SURFACE,
                         edgecolors=color if marker == "s" else SURFACE, linewidths=1.2, marker=marker,
                         label=label, zorder=3)
    for value, label in ((summary["threshold_avg_gmacs"], "official threshold 1.0"),
                         (NARROW_MACS / 1e9, "3H only")):
        index = int(torch.argmin((gmacs - value).abs()))
        ax.annotate(label, (gmacs[index], top1[index]), xytext=(6, -14), textcoords="offset points",
                    color=MUTED, fontsize=8)
    for name, place in (("drop_k0_topk_50", (2.32, 77.6)), ("drop_k0_topk_75", (1.95, 80.6))):
        row = rows[name]
        zoom.annotate(f"Drop Top-K {round(100 * row['ratio'])}%: {row['overall_top1']:.2f}\n"
                      f"(curve at same GMACs: {row['tau_matched_top1']:.2f})", (row["avg_gmacs"], row["overall_top1"]),
                      xytext=place, color=TEXT, fontsize=8, ha="left", va="center",
                      arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.75))
    zoom.set_xlim(1.9, 3.0)
    zoom.set_ylim(76.5, 81)
    zoom.set_title("Zoom: the drop points near the curve", color=TEXT, fontsize=10.5, loc="left")
    for axis in (ax, zoom):
        axis.set_xlabel("Average GMACs per image (analytic, ideal sparse kernels)", color=MUTED, fontsize=9)
    ax.set_ylabel("ImageNet-1K val Top-1 (%)", color=MUTED, fontsize=9)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=TEXT)
    ax.set_title("Experiment 3: accuracy per compute vs. routing fewer images",
                 color=TEXT, fontsize=11, loc="left")
    fig.text(0.01, -0.02, NOTE, color=MUTED, fontsize=8)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def plot_sink(sink, output):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.4), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    maps = [(sink["top_attention"], "Most-attended patch (3H CLS attention)"),
            (sink["top_norm3"], "Largest-norm patch (3H last block)")]
    vmax = max(torch.bincount(index, minlength=PATCHES).max().item() for index, _ in maps) / len(maps[0][0]) * 100
    for ax, (index, title) in zip(axes, maps):
        share = torch.bincount(index, minlength=PATCHES).double().view(14, 14) / len(index) * 100
        image = ax.imshow(share, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    bar.set_label("% of hard images where this position wins", color=MUTED, fontsize=8)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    fig.suptitle("Experiment 4: where the top token sits on the 14 x 14 patch grid", color=TEXT, fontsize=12,
                 x=0.01, ha="left")
    fig.text(0.01, -0.02, NOTE, color=MUTED, fontsize=8)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def quantiles(values):
    q = torch.quantile(values.double(), torch.tensor([0.1, 0.5, 0.9], dtype=torch.double))
    return {"p10": round(q[0].item(), 3), "median": round(q[1].item(), 3), "p90": round(q[2].item(), 3)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="evaluate_h1_diagnosis.py result directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run / "summary.json").read_text())
    data = torch.load(args.run / "per_image.pt", weights_only=True)
    curve = tau_curve(data)
    fraction = summary["routed_images"] / summary["samples"]
    summary["threshold_avg_gmacs"] = (NARROW_MACS + fraction * WIDE_FULL_MACS) / 1e9
    rows = {}
    for row in summary["results"]:
        row["wide_gmacs"] = wide_macs(row) / 1e9
        row["avg_gmacs"] = (NARROW_MACS + fraction * wide_macs(row)) / 1e9
        index = int(torch.argmin((curve[0] - row["avg_gmacs"]).abs()))
        row["tau_matched_top1"] = curve[1][index].item()
        rows[row["name"]] = row

    fields = ["name", "experiment", "mode", "k", "ratio", "policy", "overall_top1", "routed_top1",
              "wide_gmacs", "avg_gmacs", "tau_matched_top1"]
    with open(args.output / "h1_diagnosis.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fields)
        for row in rows.values():
            writer.writerow([f"{row[f]:.3f}" if isinstance(row[f], float) else row[f] for f in fields])

    plot_freeze_drop(rows, summary, args.output / "h1_freeze_vs_drop.png")
    plot_full_k(rows, summary, args.output / "h1_full_k.png")
    plot_pareto(rows, summary, curve, args.output / "h1_pareto.png")
    plot_sink(data["sink"], args.output / "h1_sink_positions.png")

    sink = data["sink"]
    attention_positions = torch.bincount(sink["top_attention"], minlength=PATCHES)
    key = {
        "gmacs": {"narrow": NARROW_MACS / 1e9, "wide_full": WIDE_FULL_MACS / 1e9,
                  "threshold_avg": summary["threshold_avg_gmacs"]},
        "accuracy": {k: summary[k] for k in ("narrow_only_top1", "wide_all_top1", "threshold_top1",
                                              "routed_narrow_top1", "routed_wide_top1", "routed_images")},
        "sink": {
            "same_top1_share": (sink["top_attention"] == sink["top_norm3"]).double().mean().item(),
            "top2_overlap_share": {n: (sink["top2_overlap"] == n).double().mean().item() for n in (0, 1, 2)},
            "attention_top2_mass": quantiles(sink["attention_top2_mass"]),
            "norm3_at_top_attention_over_median": quantiles(sink["norm3_at_top_attention"]),
            "norm3_max_over_median": quantiles(sink["norm3_max"]),
            "norm6_at_top_attention_over_median": quantiles(sink["norm6_at_top_attention"]),
            "norm6_max_over_median": quantiles(sink["norm6_max"]),
            "top_attention_top5_positions": [(divmod(int(i), 14), round(100 * attention_positions[i].item()
                                              / len(sink["top_attention"]), 2))
                                             for i in attention_positions.argsort(descending=True)[:5]],
            "top_attention_border_share": sum(attention_positions.view(14, 14)[r, c].item()
                                              for r in range(14) for c in range(14)
                                              if r in (0, 13) or c in (0, 13)) / len(sink["top_attention"]),
        },
    }
    (args.output / "h1_diagnosis_key_numbers.json").write_text(json.dumps(key, indent=2) + "\n")
    print(json.dumps(key, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
