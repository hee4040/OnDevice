#!/usr/bin/env python3
"""Analyze stage failures and inspect probabilities for routed regressions."""
import argparse
import base64
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Completed stage-analysis directory")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size or worker count")
    source = args.source.resolve()
    source_summary = json.loads((source / "summary.json").read_text())
    if not source_summary.get("complete") or not source_summary.get("full_validation"):
        parser.error("A completed full-validation stage analysis is required")
    env = json.loads((source / "environment.json").read_text())
    data_dir = Path(env["arguments"]["data_dir"])
    repo = Path(env["arguments"]["repo"])
    checkpoint = Path(env["arguments"]["checkpoint"])
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    sys.path.insert(0, str(repo))
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    import timm
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    from timm.data import create_transform, resolve_data_config
    from validate import _load_eval_checkpoint, RLDropping
    from PIL import Image

    if not Path(timm.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Official custom timm is required")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != env["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint differs from source analysis")
    revision = json.loads((data_dir / ".preparation/source.json").read_text())["revision"]
    cache = Path(os.environ["HF_HOME"]) / "hub/datasets--ILSVRC--imagenet-1k/snapshots" / revision / "data"
    parquet = next(cache.glob("validation*.parquet"))
    names = json.loads(pq.read_schema(parquet).metadata[b"huggingface"])["info"]["features"]["label"]["names"]
    if len(names) != 1000:
        raise RuntimeError("Expected 1000 dataset class names")
    rows = list(csv.DictReader((source / "images.csv").open()))
    errors = [r for r in rows if r["routed_correct"] == "0"]
    regressions = [r for r in rows if r["group"] == "regressed" and r["routed_to_6h"] == "1"]
    output = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-failure-analysis")
    output.mkdir(parents=True)
    print(f"Results: {output}", flush=True)
    classes = []
    for label in range(1000):
        subset = [r for r in rows if int(r["target"]) == label]
        eligible = [r for r in subset if r["routed_to_6h"] == "1" and r["narrow_correct"] == "1"]
        regressed = sum(r["group"] == "regressed" for r in eligible)
        classes.append({"label": label, "name": names[label], "images": len(subset),
                        "narrow_errors": sum(r["narrow_correct"] == "0" for r in subset),
                        "wide_errors": sum(r["wide_correct"] == "0" for r in subset),
                        "routed_errors": sum(r["routed_correct"] == "0" for r in subset),
                        "routed_narrow_correct": len(eligible), "routed_regressions": regressed,
                        "regression_rate_among_routed_narrow_correct": regressed / len(eligible) if eligible else None})
    classes.sort(key=lambda r: (-r["routed_errors"], r["label"]))
    write_csv(output / "class_statistics.csv", classes)

    def pairs(subset, prediction):
        counts = Counter((int(r["target"]), int(r[prediction])) for r in subset)
        return [{"target": a, "target_name": names[a], "prediction": b, "prediction_name": names[b], "count": n}
                for (a, b), n in counts.most_common()]

    regression_pairs = pairs(regressions, "wide_prediction")
    write_csv(output / "routed_error_pairs.csv", pairs(errors, "routed_prediction"))
    write_csv(output / "regression_pairs.csv", regression_pairs)
    torch.manual_seed(env["arguments"]["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model = timm.create_model("thinkingvit", pretrained=False, thinking_stages=[3, 6])
    _load_eval_checkpoint(model, str(checkpoint), use_ema=True)
    RLDropping(model, used_heads=6, max_heads=6, ema=False)
    model = model.cuda().eval()
    config = resolve_data_config({}, model=model, use_test_size=True)
    dataset = ImageFolder(data_dir / "validation", transform=create_transform(**config, is_training=False))
    indices = [int(r["dataset_index"]) for r in regressions]
    for index, row in zip(indices, regressions):
        if Path(dataset.samples[index][0]).relative_to(data_dir).as_posix() != row["image_path"]:
            raise RuntimeError("Dataset ordering differs from source")
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    logits = []
    hook = model.head.register_forward_hook(lambda module, inputs, value: logits.append(value.detach()))
    details = []
    try:
        with torch.no_grad():
            for images, targets in loader:
                logits.clear()
                model(images.cuda(), threshold=env["arguments"]["threshold"], thinking_stages=[3, 6])
                if len(logits) != 2:
                    raise RuntimeError("Unexpected classifier calls")
                narrow, wide = logits
                pn, pw = narrow.softmax(1).cpu(), wide.softmax(1).cpu()
                en, ew = model.entropy(narrow)[:, 0].cpu(), model.entropy(wide)[:, 0].cpu()
                start = len(details)
                for j, target in enumerate(targets.tolist()):
                    row = regressions[start + j]
                    wrong = int(row["wide_prediction"])
                    n5, w5 = pn[j].topk(5), pw[j].topk(5)
                    details.append({**row, "target_name": names[target], "wide_prediction_name": names[wrong],
                                    "source_predictions_reproduced": int(n5.indices[0].item() == int(row["narrow_prediction"]) and w5.indices[0].item() == wrong),
                                    "narrow_true_probability": pn[j, target].item(), "wide_true_probability": pw[j, target].item(),
                                    "narrow_wrong_probability": pn[j, wrong].item(), "wide_wrong_probability": pw[j, wrong].item(),
                                    "narrow_top1_margin": (n5.values[0] - n5.values[1]).item(),
                                    "wide_top1_margin": (w5.values[0] - w5.values[1]).item(),
                                    "wide_true_rank": int((pw[j] > pw[j, target]).sum().item()) + 1,
                                    "narrow_entropy_rerun": en[j].item(), "wide_entropy": ew[j].item(),
                                    "narrow_top5": json.dumps([(names[k], float(pn[j, k])) for k in n5.indices.tolist()]),
                                    "wide_top5": json.dumps([(names[k], float(pw[j, k])) for k in w5.indices.tolist()])})
                print(f"Regression probability analysis: {len(details)}/{len(regressions)}", flush=True)
    finally:
        hook.remove()
    write_csv(output / "regressions_detailed.csv", details)
    reproduced = [r for r in details if r["source_predictions_reproduced"]]
    if not reproduced:
        raise RuntimeError("No regression predictions reproduced")
    rank_counts = Counter(r["wide_true_rank"] for r in reproduced)
    selected = []
    # One deterministic example per leading confusion pair, then largest wrong confidence.
    for pair in regression_pairs[:12]:
        match = next((r for r in reproduced if int(r["target"]) == pair["target"] and int(r["wide_prediction"]) == pair["prediction"]), None)
        if match is not None:
            selected.append(match)
    existing = {r["dataset_index"] for r in selected}
    for row in sorted(reproduced, key=lambda r: -r["wide_wrong_probability"]):
        if row["dataset_index"] not in existing:
            selected.append(row)
            existing.add(row["dataset_index"])
        if len(selected) >= 20:
            break
    cards = []
    (output / "model_inputs").mkdir()
    for n, row in enumerate(selected):
        tensor, _ = dataset[int(row["dataset_index"])]
        mean, std = torch.tensor(config["mean"])[:, None, None], torch.tensor(config["std"])[:, None, None]
        pixels = ((tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        # Export the actual normalized model input after undoing normalization for inspection.
        im = Image.fromarray(pixels)
        png = output / "model_inputs" / f"case_{n:02d}.png"
        im.save(png)
        row["inspection_image"] = str(png.relative_to(output))
        original = data_dir / row["image_path"]
        with Image.open(original) as source_image:
            original_format = source_image.format
        mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp", "GIF": "image/gif"}.get(original_format, "image/jpeg")
        original_data = base64.b64encode(original.read_bytes()).decode()
        crop_data = base64.b64encode(png.read_bytes()).decode()
        escaped = html.escape
        cards.append(f'<article><h2>Case {n}: {escaped(row["target_name"])} → {escaped(row["wide_prediction_name"])}</h2>'
                     f'<img src="data:{mime};base64,{original_data}" alt="Original image"> '
                     f'<img src="data:image/png;base64,{crop_data}" alt="Actual 224 model input">'
                     f'<p>Original (left), actual model input (right). Dataset index: {row["dataset_index"]}.</p>'
                     f'<p>P(true): {row["narrow_true_probability"]:.3f} → {row["wide_true_probability"]:.3f}; '
                     f'P(wrong): {row["narrow_wrong_probability"]:.3f} → {row["wide_wrong_probability"]:.3f}; '
                     f'true class rank in 6H: {row["wide_true_rank"]}</p>'
                     f'<details><summary>Top-5 probabilities</summary><pre>3H: {escaped(row["narrow_top5"])}\n6H: {escaped(row["wide_top5"])}</pre></details></article>')
    (output / "gallery.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>ThinkingViT regression cases</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px}article{border-top:1px solid #ccc;padding:20px 0}img{max-width:45%;height:300px;object-fit:contain}pre{white-space:pre-wrap}h2{font-size:20px}</style>'
        '<h1>3H correct → 6H incorrect: diagnostic examples</h1><p>Selected by leading confusion pairs, then highest wrong-class confidence. '
        'Examples are not a random or exhaustive visual sample. Probabilities were rerun in regression-only batches; prediction reproduction is checked. '
        'Dataset labels are evaluation targets, not a claim that alternate descriptions are visually impossible.</p>' + ''.join(cards) + '</html>')
    write_json(output / "selected_cases.json", selected)
    summary = {
        "source_run": str(source.relative_to(ROOT)), "samples": len(rows), "routed_errors": len(errors),
        "early_exit_errors": sum(r["routed_to_6h"] == "0" for r in errors),
        "routed_regressions": len(regressions),
        "source_regressions_true_in_wide_top5": sum(r["wide_top5_correct"] == "1" for r in regressions),
        "probability_rerun_predictions_reproduced": len(reproduced),
        "probability_rerun_predictions_changed": len(details) - len(reproduced),
        "reproduced_true_rank_histogram": {str(k): v for k, v in sorted(rank_counts.items())},
        "reproduced_wrong_probability_ge_0_5": sum(r["wide_wrong_probability"] >= .5 for r in reproduced),
        "reproduced_wrong_probability_ge_0_9": sum(r["wide_wrong_probability"] >= .9 for r in reproduced),
        "median_narrow_top1_margin": float(np.median([r["narrow_top1_margin"] for r in reproduced])),
        "median_wide_top1_margin": float(np.median([r["wide_top1_margin"] for r in reproduced])),
        "worst_classes": classes[:15], "top_regression_pairs": regression_pairs[:15],
        "checkpoint_sha256": env["checkpoint_sha256"], "batch_size": args.batch_size,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "precision": env["precision"],
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Class statistics use original full-validation records. Probability rerun uses only routed regressions; "
                "batch composition differs from the source and reproduction is reported explicitly. "
                "No automatic visual cause labels are assigned. 50 validation images per class."}
    write_json(output / "summary.json", summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in {"worst_classes", "top_regression_pairs"}}, indent=2), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
