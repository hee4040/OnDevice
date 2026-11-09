#!/usr/bin/env python3
"""Experiment 2 analysis: GPU time per runtime, float16 accuracy, drop gain at equal GPU time.

Time of one image at threshold 1.0 comes from batch-1 stage medians: 3H, plus 6H for routed
images (official), or 3H with attention plus drop 6H (drop). Equal-time gain: the official
threshold curve is moved to the same time (more or fewer images to 6H, by entropy) and compared
image by image with the fine-tuned drop model in the same precision; bootstrap 95% interval.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from analyze_followup import bootstrap, drop_macs
from plot_h1_diagnosis import WIDE_FULL_MACS

PERCENTS = [25, 50]
RUNTIMES = {"eager": "기본 실행", "CUDA Graphs": "CUDA Graphs", "torch.compile": "torch.compile",
            "torch.compile + CUDA Graphs": "compile\n+ CUDA Graphs"}
plt.rcParams.update({"font.family": ["Malgun Gothic", "NanumGothic", "DejaVu Sans"], "axes.unicode_minus": False})
BLUE, ORANGE, AQUA, TEXT, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
NOTE = "{gpu} · 이미지 1장씩(batch 1) · 단계별 시간 중앙값으로 계산 · τ = 1.0"


def style(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def top1(data, key, routed):
    return 100 * torch.where(routed, data[key], data["narrow_correct"]).double().mean().item()


def at_time(model, reference, t, percent):
    """Fine-tuned drop at threshold 1.0 against the official curve at the same batch-1 GPU time."""
    routed = model["entropy"] >= 1.0
    correct = torch.where(routed, model[f"drop_{percent}_correct"], model["narrow_correct"])
    ms = t["3H + attention"] + routed.double().mean().item() * t[f"6H drop Top-K {percent}%"]
    share = min(max((ms - t["3H"]) / t["6H full"], 0.0), 1.0)
    hard = torch.zeros_like(routed)
    hard[torch.argsort(reference["entropy"], descending=True)[:round(share * len(hard))]] = True
    baseline = torch.where(hard, reference["wide_correct"], reference["narrow_correct"])
    difference = correct.double() - baseline.double()
    return {"ratio": percent, "ms": ms, "top1": 100 * correct.double().mean().item(),
            "official_share_same_ms": share, "official_top1_same_ms": 100 * baseline.double().mean().item(),
            "gain": 100 * difference.mean().item(), "gain_ci95": bootstrap(difference)}


def plot_time(key, output):
    labels = [label for label in key["pipeline_ms"]]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    style(ax)
    width = 0.26
    for i, (name, color, legend) in enumerate((("official", TEXT, "원래 모델, 6H 전부 계산"),
                                               ("drop_25", BLUE, "토큰 제거 Top-K 25% (파인튜닝)"),
                                               ("drop_50", AQUA, "토큰 제거 Top-K 50% (파인튜닝)"))):
        values = [key["pipeline_ms"][label][name] for label in labels]
        ax.bar([j + (i - 1) * width for j in range(len(labels))], values, width, color=color, label=legend)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([("FP32" if l.startswith("float32") else "FP16") + "\n" + RUNTIMES[l.split(" | ")[1]]
                        for l in labels], fontsize=8)
    ax.set_ylabel("이미지 1장 평균 시간 (ms)", color=MUTED, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.text(0.01, -0.06, NOTE.format(gpu=key["gpu"]), color=MUTED, fontsize=8)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_batch(key, output):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    for ax, percent in zip(axes, PERCENTS):
        style(ax)
        for (setting, values), color in zip(key["batch_drop_share"].items(), (TEXT, BLUE, AQUA)):
            batches = sorted(values[percent])
            ax.plot(batches, [values[percent][b] for b in batches], color=color, marker="o", ms=4, lw=1.5,
                    label=setting)
        theory = 100 * drop_macs(percent) / WIDE_FULL_MACS
        ax.axhline(theory, color=MUTED, lw=0.8, ls="--")
        ax.text(1, theory + 2, f"연산량 비율 {theory:.0f}%", color=MUTED, fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 8, 32], ["1", "8", "32"])
        ax.set_title(f"토큰 제거 Top-K {percent}%", color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("한 번에 처리하는 이미지 수 (batch)", color=MUTED, fontsize=9)
    axes[0].set_ylabel("6H 전부 계산 대비 시간 (%)", color=MUTED, fontsize=9)
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.01, -0.08, f"{key['gpu']} · 6H 단계 시간 중앙값 · 100%면 토큰을 빼도 시간이 그대로", color=MUTED, fontsize=8)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency", type=Path, required=True, help="optimize_latency.py run")
    for name in ("official", "finetuned", "official-fp16", "finetuned-fp16"):
        parser.add_argument(f"--{name}", type=Path, required=True, help="evaluate_drop.py run")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    load = lambda run: torch.load(run / "per_image.pt", weights_only=True)
    data = {"float32": (load(args.official), load(args.finetuned)),
            "float16": (load(args.official_fp16), load(args.finetuned_fp16))}
    key = {"accuracy": {}, "fp16_changes": {}, "stage_ms": {}, "errors": [], "pipeline_ms": {}, "equal_time": {},
           "batch_drop_share": {}}
    for precision, models in data.items():
        for name, d in zip(("official", "fine_tuned"), models):
            routed = d["entropy"] >= 1.0
            key["accuracy"][f"{precision} {name}"] = {
                "routed": int(routed.sum()), "3h": 100 * d["narrow_correct"].double().mean().item(),
                "threshold_1_full_6h": top1(d, "wide_correct", routed),
                **{f"threshold_1_drop_{p}": top1(d, f"drop_{p}_correct", routed) for p in PERCENTS}}
    for i, name in enumerate(("official", "fine_tuned")):
        a, b = data["float32"][i], data["float16"][i]
        key["fp16_changes"][name] = {"routing": int(((a["entropy"] >= 1) != (b["entropy"] >= 1)).sum()), **{
            k: int((a[f"{k}_correct"] != b[f"{k}_correct"]).sum()) for k in ("narrow", "wide", "drop_25", "drop_50")}}
    latency = json.loads((args.latency / "summary.json").read_text())
    rows, key["gpu"] = latency["rows"], latency["environment"]["gpu"]
    for row in rows:
        if "error" in row:
            key["errors"].append(row)
        else:
            label = f'{row["precision"]} | {row["runtime"]} | batch {row["batch"]}'
            key["stage_ms"].setdefault(label, {})[row["stage"]] = row["median_ms"]
    for label, t in key["stage_ms"].items():
        precision, runtime, batch = label.split(" | ")
        if len(t) < 6:  # a stage failed to compile or capture
            continue
        if batch == "batch 1":
            official, finetuned = data[precision]
            share = (official["entropy"] >= 1.0).double().mean().item()
            key["equal_time"][label] = [at_time(finetuned, official, t, p) for p in PERCENTS]
            key["pipeline_ms"][label] = {"official": t["3H"] + share * t["6H full"],
                                         **{f"drop_{r['ratio']}": r["ms"] for r in key["equal_time"][label]}}
        setting = ("FP32 " if precision == "float32" else "FP16 ") + RUNTIMES[runtime].replace("\n", " ")
        for p in PERCENTS:
            key["batch_drop_share"].setdefault(setting, {}).setdefault(p, {})[int(batch.split()[1])] = (
                100 * t[f"6H drop Top-K {p}%"] / t["6H full"])
    key["batch_drop_share"] = {s: v for s, v in key["batch_drop_share"].items() if len(v[PERCENTS[0]]) > 1}
    (args.output / "exp2_key_numbers.json").write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n",
                                                        encoding="utf-8")
    plot_time(key, args.output / "exp2_time.png")
    plot_batch(key, args.output / "exp2_batch.png")
    print(json.dumps({k: key[k] for k in ("accuracy", "fp16_changes", "pipeline_ms", "equal_time")}, indent=1,
                     ensure_ascii=False))
    print(f"{len(key['errors'])} settings failed")


if __name__ == "__main__":
    main()
