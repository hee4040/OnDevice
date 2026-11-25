"""Analytic executed-shape MAC counts and accuracy/compute operating points.

Counts dense matmul/conv MACs and duplicate CLS-query matmuls explicitly.
No claim about memory traffic, softmax/topk/norm work, or GPU execution time.
"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATCHES, TOKENS, NARROW, WIDE, DEPTH, CLASSES = 196, 197, 192, 384, 12, 1000


def block_macs(tokens, width):
    return 12 * tokens * width ** 2 + 2 * tokens ** 2 * width


def macs(method, ratio=1.0):
    if method not in ("full", "last_cls", "global_cls", "evo_layer", "evo_stage") or not 0 < ratio <= 1:
        raise ValueError("Invalid method or ratio")
    # patch_embed is the same 384-channel Conv2d in BOTH passes; the first
    # block's LayerNorm slices the narrow input to 192 channels afterwards.
    patch = PATCHES * 3 * 16 * 16 * WIDE
    narrow = patch + DEPTH * block_macs(TOKENS, NARROW) + NARROW * CLASSES
    n = TOKENS if method == "full" else 1 + round(PATCHES * ratio)
    wide = patch + TOKENS * NARROW * WIDE + DEPTH * block_macs(n, WIDE) + WIDE * CLASSES
    narrow_capture = 0 if method == "full" else (1 if method == "last_cls" else DEPTH)
    narrow_extra = narrow_capture * TOKENS * NARROW
    wide_extra = 0
    if method.startswith("evo_") and ratio < 1:
        # One representative uses one of the patch-budget slots.
        placeholder_count = PATCHES - (n - 2)
        wide_extra = DEPTH * (n * WIDE + placeholder_count * WIDE)
    return {"narrow_core": narrow, "wide_core": wide,
            "narrow_selection": narrow_extra, "wide_selection_aggregation": wide_extra,
            "wide_block_tokens": n}


def average_macs(method, ratio, routed_fraction):
    if not 0 <= routed_fraction <= 1:
        raise ValueError("Invalid routed fraction")
    cost = macs(method, ratio)
    core = cost["narrow_core"] + routed_fraction * cost["wide_core"]
    auxiliary = cost["narrow_selection"] + routed_fraction * cost["wide_selection_aggregation"]
    return core / 1e9, (core + auxiliary) / 1e9


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def compute_summary(directory):
    old_study = read(ROOT / "results/drop-ft-20261106/study.json") or {}
    official_dir = Path(old_study.get("official_eval", ROOT / "results/20261106T124010.390968Z-drop-eval-official"))
    official = read(official_dir / "summary.json")
    if not official:
        return {"rows": [], "status": "Official reference missing"}
    fraction = official["routed_at_threshold_1"] / 50000
    core, total = average_macs("full", 1, fraction)
    baseline = {"model": "official", "method": "full", "ratio": 1.0,
                "top1": official["top1"]["wide_threshold_1"], "gmacs_core": core, "gmacs": total,
                "routed_fraction": fraction, "phase": "trained", "samples": 50000}
    rows = [baseline]
    inputs = [("existing_kd", ROOT / "results/drop-ft-20261106/kd/eval/summary.json")]
    inputs += [(name, directory / name / "eval/summary.json")
               for name in ("global_cls", "evo_schedule", "global_to_evo")]
    for name, path in inputs:
        result = read(path)
        if not result:
            continue
        fraction = result.get("routed", result.get("routed_at_threshold_1", 0)) / result.get("samples", 50000)
        for ratio in (.25, .5, .75):
            methods = ["last_cls"] if name == "existing_kd" else (
                ["global_cls"] if name == "global_cls" else ["evo_layer", "evo_stage"])
            for method in methods:
                if name == "existing_kd":
                    top1 = result["top1"].get(f"drop_{round(100 * ratio)}_threshold_1")
                else:
                    top1 = result["top1"].get(f"{method}_{round(100 * ratio)}_correct", {}).get("threshold_1")
                if top1 is None:
                    continue
                core, total = average_macs(method, ratio, fraction)
                rows.append({"model": name, "method": method, "ratio": ratio, "top1": top1,
                    "gmacs_core": core, "gmacs": total, "routed_fraction": fraction,
                    "phase": "trained", "samples": result.get("samples", 50000)})
    timing = read(directory / "latency-clean-final.json")
    if not timing or not timing.get("clean"):
        timing = read(directory / "latency-clean-now.json") or {}
    for row in rows:
        row["accuracy_delta_pp"] = row["top1"] - baseline["top1"]
        row["mac_reduction_percent"] = 100 * (1 - row["gmacs"] / baseline["gmacs"])
        row["non_dominated_among_listed_points"] = not any(
            other["top1"] >= row["top1"] and other["gmacs"] <= row["gmacs"] and
            (other["top1"] > row["top1"] or other["gmacs"] < row["gmacs"]) for other in rows)
        row["latency_ms"] = None
        if row["ratio"] == .5 or row["model"] == "official":
            item = timing.get("aggregate", {}).get(f"{row['model']}/{row['method']}", {}) if timing.get("clean") else {}
            row["latency_ms"] = item.get("median_mean_ms")
    return {"status": "computed", "convention": "1 MAC = multiply+accumulate, usually 2 FLOPs; GMACs per image",
        "baseline": "official full-token dynamic 3H->6H at tau=1", "rows": rows,
        "included": ["384-channel patch embedding in both passes", "QKV/projection/MLP matmuls",
                     "QK and attention-V matmuls", "recycling projection and heads",
                     "duplicate CLS-query QK products", "slow-fast weighted representative aggregation"],
        "excluded": ["norm/softmax/GELU", "bias and residual/EMA elementwise ops", "sort/topk/gather/scatter",
                     "memory traffic and CPU kernel launch/synchronization overhead"],
        "note": "Earlier repository estimates assumed a 192-channel narrow patch embedding; actual code computes 384 before slicing. These counts correct that assumption for every method."}


def write_compute(directory):
    result = compute_summary(directory)
    target = directory / "compute_tradeoff.json"
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(target)
    return result


def plot_tradeoff(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    torch.set_num_threads(1)
    result = write_compute(directory)
    study = read(ROOT / "results/drop-ft-20261106/study.json")
    data = torch.load(Path(study["official_eval"]) / "per_image.pt", map_location="cpu", weights_only=True)
    narrow, wide = data["narrow_correct"].double(), data["wide_correct"].double()
    order = torch.argsort(data["entropy"], descending=True)
    correct = narrow.sum() + torch.cat((torch.zeros(1), (wide - narrow)[order].cumsum(0)))
    fraction = torch.arange(len(narrow) + 1) / len(narrow)
    cost = macs("full")
    curve_x = (cost["narrow_core"] + fraction * cost["wide_core"]) / 1e9
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve_x.numpy(), (100 * correct / len(narrow)).numpy(), color="0.5", lw=1,
            label="Original full tokens: threshold sweep (exploratory)")
    labels = {"official": "Original, full tokens, tau=1", "existing_kd": "Original KD + drop",
              "global_cls": "Global CLS KD", "evo_schedule": "Global CLS + slow-fast KD",
              "global_to_evo": "Global CLS -> slow-fast KD (6 epochs)"}
    for name, label in labels.items():
        points = sorted([r for r in result["rows"] if r["model"] == name and r["method"] != "evo_layer"], key=lambda r:r["gmacs"])
        if not points:
            continue
        ax.plot([p["gmacs"] for p in points], [p["top1"] for p in points], "o-", label=label)
        for point in points:
            if name != "official":
                ax.annotate(f"{point['ratio']:.0%}", (point["gmacs"], point["top1"]), xytext=(3, 6), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Average analytic GMACs / image (lower is better)", ylabel="ImageNet validation Top-1 (%)",
           title="Accuracy versus arithmetic work; GPU latency is a separate measurement")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "accuracy_compute_tradeoff.png", dpi=180)
    plt.close(fig)
    (directory / "accuracy_compute_tradeoff.models.json").write_text(json.dumps(sorted(set(r["model"] for r in result["rows"]))) + "\n")


if __name__ == "__main__":
    import sys
    directory = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results/evo-vit-20261115"
    plot_tradeoff(directory)
    print(json.dumps(compute_summary(directory), indent=2))
