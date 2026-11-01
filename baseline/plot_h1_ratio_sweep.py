#!/usr/bin/env python3
"""H1 선택적 갱신 평가 결과(summary.json)를 비율별로 모아 CSV와 그래프로 만든다.

같은 (ratio, policy, seed)가 여러 실행에 있으면 뒤에 준 실행의 값을 쓴다.
ratio 0과 1은 정책과 무관하게 같은 경로이므로 모든 정책 선의 양 끝점으로 함께 그린다.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = {
    "selective": "Selective refinement (attention + MLP on selected tokens)",
    "mlp": "Full attention + selective MLP",
}
POLICIES = [  # (policy, legend, color) - 고정 순서, 색맹 검증한 3색
    ("topk", "CLS attention Top-K", "#2a78d6"),
    ("random", "Random-K (5 seeds, mean ± 1 SD)", "#1baf7a"),
    ("bottomk", "CLS attention Bottom-K", "#eb6834"),
]
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load(directories):
    rows = {}
    for directory in directories:
        for row in json.loads((Path(directory) / "summary.json").read_text())["results"]:
            rows[(row["ratio"], row["policy"], row["random_seed"])] = row["overall_top1"]
    grouped = {}
    for (ratio, policy, _), top1 in rows.items():
        grouped.setdefault((ratio, policy), []).append(top1)
    return {key: (statistics.mean(v), statistics.pstdev(v), len(v)) for key, v in sorted(grouped.items())}


def series(data, policy, low, high):
    points = []
    for (ratio, row_policy), (mean, std, _) in data.items():
        if low <= ratio <= high and (row_policy == policy or (ratio in (0.0, 1.0) and row_policy == "topk")):
            points.append((100 * ratio, mean, std))
    return sorted(points)


def plot(results, output, low, high, title, marker_step=1):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, facecolor=SURFACE)
    full = results["selective"][(1.0, "topk")][0]
    for ax, (method, data) in zip(axes, results.items()):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.75)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.axhline(full, color=MUTED, linewidth=0.75)
        ax.text(100 * low + 0.01 * (100 * high - 100 * low), full + 0.4,
                f"100% refined = full 6H ({full:.2f})", color=MUTED, fontsize=8, va="bottom")
        for policy, label, color in POLICIES:
            points = series(data, policy, low, high)
            x, y, std = zip(*points)
            if policy == "random":
                ax.fill_between(x, [a - s for a, s in zip(y, std)], [a + s for a, s in zip(y, std)],
                                color=color, alpha=0.18, linewidth=0)
            ax.plot(x, y, color=color, linewidth=1.5, marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1, solid_joinstyle="round", label=label,
                    markevery=[round(v) % marker_step == 0 for v in x])
        ax.set_title(METHODS[method], color=TEXT, fontsize=10.5, loc="left")
        ax.set_xlabel("Refined patch tokens (% of 196)", color=MUTED, fontsize=9)
        ax.set_xlim(100 * low - 0.5, 100 * high + 0.5)
    axes[0].set_ylabel("ImageNet-1K val Top-1 (%)", color=MUTED, fontsize=9)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               labelcolor=TEXT, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(title, color=TEXT, fontsize=12, x=0.01, ha="left", y=1.07)
    fig.text(0.01, -0.04, "RTX 4090 · threshold 1.0 · 50,000 validation images · dense-mask accuracy oracle "
             "(no FLOPs or latency reduction)", color=MUTED, fontsize=8)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selective", nargs="+", required=True, help="evaluate_selective.py 결과 디렉터리")
    parser.add_argument("--mlp", nargs="+", required=True, help="evaluate_mlp_selective.py 결과 디렉터리")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {"selective": load(args.selective), "mlp": load(args.mlp)}

    with open(args.output / "h1_ratio_sweep.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "ratio", "policy", "top1_mean", "top1_std", "runs"])
        for method, data in results.items():
            for (ratio, policy), (mean, std, runs) in data.items():
                writer.writerow([method, ratio, policy, f"{mean:.3f}", f"{std:.3f}", runs])

    plot(results, args.output / "h1_ratio_sweep_full.png", 0.0, 1.0,
         "H1 selective refinement: Top-1 vs. refined token ratio (0-100%)", marker_step=5)
    plot(results, args.output / "h1_ratio_sweep_76_100.png", 0.75, 1.0,
         "H1 selective refinement: Top-1 vs. refined token ratio (75-100%, 1% steps)")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
