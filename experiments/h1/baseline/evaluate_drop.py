#!/usr/bin/env python3
"""Per-image 3H, full-6H and drop Top-K correctness for every validation image.

Every image runs every variant, so any threshold can be applied afterwards. The output
feeds analyze_followup.py: the compute-matched threshold curve, bootstrap and split-half
checks of the drop gain, and the same numbers for a fine-tuned model (--weights).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_imagenet import check_counts
from evaluate_h1_diagnosis import wide_blocks, wide_input
from selective_refinement import forward_narrow_with_attention, make_update_mask

STAGES = (3, 6)
NARROW_P, WIDE_P = 0.5, 1.0
RATIOS = [0.1, 0.25, 0.5, 0.75, 0.9]


def load_model(repo, checkpoint, weights=None, device="cuda"):
    """Official EMA checkpoint, optionally overwritten by a fine-tuned state dict."""
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    sys.path.insert(0, str(repo))
    import timm
    from validate import _load_eval_checkpoint, RLDropping
    if not Path(timm.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("Official custom timm is required")
    model = timm.create_model("thinkingvit", pretrained=False, thinking_stages=STAGES)
    _load_eval_checkpoint(model, str(checkpoint), use_ema=True)
    RLDropping(model, used_heads=6, max_heads=6, ema=False)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    if model.patch_embed.num_patches != 196 or model.num_prefix_tokens != 1 or model._stage_ratio(3) != NARROW_P:
        raise RuntimeError("Expected 196 patches, one CLS token and 3H = half width")
    return model.to(device).eval()


def wide_logits(model, x0, mask=None):
    """6H logits from the 6H input; with a mask, tokens outside it leave the sequence."""
    model.p = WIDE_P
    x = wide_blocks(model, x0, WIDE_P, mask, "drop") if mask is not None else wide_blocks(model, x0, WIDE_P)
    return model.forward_head(model.norm(x, int(WIDE_P * model.embed_dim)))


def add_paths(parser):
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/imagenet1k")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/thinkingvit_3h6h_ema.pth")
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party/ThinkingViT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    parser.add_argument("--weights", type=Path, help="Fine-tuned state dict from finetune_drop.py")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="float16 weights and inputs (experiment 2)")
    args = parser.parse_args()
    check_counts(args.data_dir / "validation", 50000)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # The entropy's top-10 probabilities miss the Simplex check's tolerance in float16; values are unchanged.
    torch.distributions.Distribution.set_default_validate_args(False)
    model = load_model(args.repo.resolve(), args.checkpoint.resolve(), args.weights)
    dtype = torch.float16 if args.fp16 else torch.float32
    model = model.to(dtype)
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    from timm.data import create_transform, resolve_data_config
    config = resolve_data_config({}, model=model, use_test_size=True)
    dataset = ImageFolder(args.data_dir / "validation", transform=create_transform(**config, is_training=False))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed)).tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True)
    tag = ("finetuned" if args.weights else "official") + ("-fp16" if args.fp16 else "")
    output = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-drop-eval-{tag}")
    output.mkdir(parents=True)
    print(f"Results: {output}", flush=True)
    keys = ["narrow_correct", "wide_correct"] + [f"drop_{round(100 * r)}_correct" for r in RATIOS]
    per_image = {key: [] for key in ["entropy"] + keys}
    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader, 1):
            images, targets = images.to("cuda", dtype, non_blocking=True), targets.cuda(non_blocking=True)
            features, logits, attention = forward_narrow_with_attention(model, images, STAGES)
            x0 = wide_input(model, images, features)
            outputs = [logits, wide_logits(model, x0)] + [
                wide_logits(model, x0, make_update_mask(attention, r, "topk")) for r in RATIOS]
            if batch_index == 1:  # the full path must equal the official two-stage forward
                model.p = WIDE_P
                reference = model.forward_head(model.forward_features(images, WIDE_P, features, STAGES))
                torch.testing.assert_close(outputs[1], reference, **({"rtol": 1e-2, "atol": 1e-2} if args.fp16
                                                                     else {"rtol": 1e-5, "atol": 1e-6}))
            per_image["entropy"].append(model.entropy(logits)[:, 0].float().cpu())
            for key, value in zip(keys, outputs):
                per_image[key].append((value.argmax(1) == targets).cpu())
            if batch_index % 100 == 0:
                print(f"Evaluated {batch_index * args.batch_size}/50000", flush=True)
    per_image = {key: torch.cat(value) for key, value in per_image.items()}
    if len(per_image["entropy"]) != 50000:
        raise RuntimeError("Incomplete evaluation")
    torch.save(per_image, output / "per_image.pt")
    routed = per_image["entropy"] >= 1.0
    summary = {"model": tag, "weights": str(args.weights) if args.weights else None,
               "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest() if args.weights else None,
               "routed_at_threshold_1": int(routed.sum()), "gpu": torch.cuda.get_device_name(),
               "precision": "float16 weights and inputs" if args.fp16 else "float32, TF32 allowed (official default)",
               "full_path_matches_official": True, "top1": {}}
    for key in keys:
        summary["top1"][key.removesuffix("_correct") + "_all"] = 100 * per_image[key].double().mean().item()
        if key != "narrow_correct":  # threshold 1.0: easy images keep the 3H answer
            mixed = torch.where(routed, per_image[key], per_image["narrow_correct"])
            summary["top1"][key.removesuffix("_correct") + "_threshold_1"] = 100 * mixed.double().mean().item()
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
