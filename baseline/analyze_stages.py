#!/usr/bin/env python3
"""Record per-image narrow/wide predictions and narrow final-block CLS attention."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_imagenet import check_counts

GROUPS = ("both_correct", "corrected", "regressed", "both_wrong")


def transition_group(narrow_correct, wide_correct):
    if narrow_correct:
        return "both_correct" if wide_correct else "regressed"
    return "corrected" if wide_correct else "both_wrong"


def summarize(rows):
    result = {}
    for name, subset in (
        ("all", rows),
        ("routed_to_6h", [r for r in rows if r["routed_to_6h"]]),
        ("exited_at_3h", [r for r in rows if not r["routed_to_6h"]]),
    ):
        counts = Counter(r["group"] for r in subset)
        n = len(subset)
        result[name] = {
            "images": n,
            "groups": {group: counts[group] for group in GROUPS},
            "narrow_top1": 100 * sum(r["narrow_correct"] for r in subset) / n if n else None,
            "wide_top1": 100 * sum(r["wide_correct"] for r in subset) / n if n else None,
            "routed_top1": 100 * sum(r["routed_correct"] for r in subset) / n if n else None,
            "net_corrected_images": counts["corrected"] - counts["regressed"],
        }
    return result


def cls_attention_from_qkv(attention, projected, heads):
    """Reconstruct only the CLS row; leave the model's fused attention unchanged."""
    batch, tokens, _ = projected.shape
    qkv = projected.reshape(batch, tokens, 3, heads, attention.head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    q, k = attention.q_norm(q[:, :, :1]), attention.k_norm(k)
    probabilities = ((q * attention.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    return probabilities.mean(dim=1).squeeze(1)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/imagenet1k")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/thinkingvit_3h6h_ema.pth")
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party/ThinkingViT")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--samples", type=int, default=50000, help="Use 50,000 for the full validation report")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.samples <= 50000 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid samples, batch-size or workers")
    args.repo, args.data_dir, args.checkpoint = args.repo.resolve(), args.data_dir.resolve(), args.checkpoint.resolve()
    check_counts(args.data_dir / "validation", 50000)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    sys.path.insert(0, str(args.repo))
    import numpy as np
    import torch
    import timm
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    from timm.data import create_transform, resolve_data_config
    from validate import _load_eval_checkpoint, RLDropping

    if not Path(timm.__file__).resolve().is_relative_to(args.repo):
        raise RuntimeError("Official custom timm is required")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    model = timm.create_model("thinkingvit", pretrained=False, thinking_stages=[3, 6])
    _load_eval_checkpoint(model, str(args.checkpoint), use_ema=True)
    RLDropping(model, used_heads=6, max_heads=6, ema=False)
    model = model.cuda().eval()
    config = resolve_data_config({}, model=model, use_test_size=True)
    dataset = ImageFolder(args.data_dir / "validation", transform=create_transform(**config, is_training=False))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.samples].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, shuffle=False)
    output = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-stage-analysis")
    output.mkdir(parents=True)
    print(f"Results: {output}", flush=True)
    write_json(output / "environment.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "precision": "float32, AMP off, TF32 allowed", "preprocessing": config,
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "repo_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "attention": "Head-mean CLS query attention in the final 3H block, before dropout; "
                     "column 0 = CLS self, columns 1..196 = row-major 14x14 patch grid after preprocessing; "
                     "patch weights are not renormalized after excluding CLS.",
        "execution": "Offline diagnosis executes BOTH stages for every image. routed_to_6h records "
                     "the existing entropy gate decision, not whether this diagnostic ran the wide stage.",
    })
    write_json(output / "sample_indices.json", indices)
    tokens = model.patch_embed.num_patches + model.num_prefix_tokens
    if tokens != 197 or model.num_prefix_tokens != 1:
        raise RuntimeError("This analysis expects a 14x14 patch grid and one CLS token")
    attention_array = np.lib.format.open_memmap(output / "narrow_cls_attention.partial.npy", mode="w+",
                                               dtype=np.float32, shape=(args.samples, tokens))
    captured_logits, captured_attention = [], []
    attention = model.blocks[-1].attn

    def capture_logits(module, inputs, value):
        captured_logits.append(value.detach())

    def capture_attention(module, inputs, projected):
        heads = projected.shape[-1] // (3 * attention.head_dim)
        if heads == 3:
            captured_attention.append(cls_attention_from_qkv(attention, projected.detach(), heads))

    head_hook = model.head.register_forward_hook(capture_logits)
    attention_hook = attention.qkv.register_forward_hook(capture_attention)
    records = []
    fields = ["attention_row", "dataset_index", "image_path", "target", "narrow_prediction", "wide_prediction",
              "narrow_correct", "wide_correct", "narrow_top5_correct", "wide_top5_correct", "narrow_entropy",
              "routed_to_6h", "routed_prediction", "routed_correct", "group"]
    try:
        with torch.no_grad(), (output / "images.partial.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for batch_index, (images, targets) in enumerate(loader, 1):
                captured_logits.clear()
                captured_attention.clear()
                x = images.cuda(non_blocking=True)
                # Check on the first batch that observation hooks do not change predictions.
                if batch_index == 1:
                    head_hook.remove()
                    attention_hook.remove()
                    reference, reference_stage = model(x, threshold=args.threshold, thinking_stages=[3, 6])
                    head_hook = model.head.register_forward_hook(capture_logits)
                    attention_hook = attention.qkv.register_forward_hook(capture_attention)
                routed_logits, stage = model(x, threshold=args.threshold, thinking_stages=[3, 6])
                if batch_index == 1:
                    torch.testing.assert_close(routed_logits, reference, rtol=0, atol=0)
                    torch.testing.assert_close(stage, reference_stage, rtol=0, atol=0)
                    del reference, reference_stage
                if len(captured_logits) != 2 or len(captured_attention) != 1:
                    raise RuntimeError("Unexpected official stage/head execution")
                narrow, wide = captured_logits
                entropy = model.entropy(narrow)
                torch.testing.assert_close(stage, (~(entropy < args.threshold)).long(), rtol=0, atol=0)
                torch.testing.assert_close(routed_logits, torch.where(stage.bool(), wide, narrow), rtol=0, atol=0)
                weights = captured_attention[0]
                if not torch.isfinite(weights).all() or (weights < 0).any():
                    raise RuntimeError("Invalid attention probabilities")
                torch.testing.assert_close(weights.sum(1), torch.ones(len(images), device=x.device), rtol=1e-5, atol=1e-6)
                start = len(records)
                attention_array[start:start + len(images)] = weights.cpu().numpy()
                narrow_top5, wide_top5 = narrow.topk(5, dim=1).indices.cpu().tolist(), wide.topk(5, dim=1).indices.cpu().tolist()
                entropy_values, route_values = entropy[:, 0].cpu().tolist(), stage[:, 0].cpu().tolist()
                for j, target in enumerate(targets.tolist()):
                    idx = indices[start + j]
                    narrow_correct, wide_correct = narrow_top5[j][0] == target, wide_top5[j][0] == target
                    routed_prediction = wide_top5[j][0] if route_values[j] else narrow_top5[j][0]
                    row = dict(zip(fields, [
                        start + j, idx, str(Path(dataset.samples[idx][0]).relative_to(args.data_dir)), target,
                        narrow_top5[j][0], wide_top5[j][0], int(narrow_correct), int(wide_correct),
                        int(target in narrow_top5[j]), int(target in wide_top5[j]), entropy_values[j],
                        route_values[j], routed_prediction, int(routed_prediction == target),
                        transition_group(narrow_correct, wide_correct),
                    ]))
                    writer.writerow(row)
                    records.append(row)
                if batch_index % 10 == 0 or len(records) == args.samples:
                    handle.flush()
                    print(f"Analyzed {len(records)}/{args.samples}", flush=True)
                del x, routed_logits, stage, narrow, wide, entropy, weights
    finally:
        head_hook.remove()
        attention_hook.remove()
        attention_array.flush()
    if len(records) != args.samples:
        raise RuntimeError("Incomplete analysis")
    del attention_array
    (output / "narrow_cls_attention.partial.npy").rename(output / "narrow_cls_attention.npy")
    (output / "images.partial.csv").rename(output / "images.csv")
    report = {"complete": True, "samples": len(records), "full_validation": len(records) == 50000,
              "threshold": args.threshold, "hook_output_equality_verified_on_first_batch": True,
              "groups": summarize(records)}
    write_json(output / "summary.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
