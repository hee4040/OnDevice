#!/usr/bin/env python3
"""Diagnose why H1 selective refinement collapses, in one validation pass.

All variants change only the 6H pass of images routed by the threshold; other
images keep the 3H answer, as in the H1 evaluators.
  1. freeze vs drop: unselected tokens stay as stale K/V (H1) or leave the 6H sequence.
  2. full-k: every token is updated in the first k 6H blocks, then freeze or drop.
  3. tau curve: per-image 3H entropy and 3H/6H correctness for a compute-matched baseline.
  4. sink check: 3H CLS attention vs 3H final-block token norms, and an MLP-selective
     policy that skips the highest-norm tokens instead of the highest-attention ones.
Drop really shortens the sequence; freeze and MLP-selective stay dense-mask oracles.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_imagenet import check_counts

MODES = ("freeze", "drop")
RATIOS = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
FULL_K = [2, 4, 6, 8, 10]
FULL_K_RATIOS = [0.25, 0.5, 0.75]
MLP_RATIOS = [0.25, 0.5, 0.75, 0.9, 0.95, 0.97, 0.99]
MLP_HIGH = [0.9, 0.95, 0.97, 0.99]


def experiment_configs():
    rows = [dict(experiment="freeze_drop", mode=m, k=0, ratio=0.0, policy="topk") for m in MODES]
    rows += [dict(experiment="freeze_drop", mode=m, k=0, ratio=r, policy=p)
             for r in RATIOS for p in ("topk", "bottomk", "random") for m in MODES]
    rows += [dict(experiment="full_k", mode=m, k=k, ratio=r, policy="topk")
             for k in FULL_K for r in FULL_K_RATIOS for m in MODES]
    rows += [dict(experiment="full_k", mode=m, k=k, ratio=0.5, policy="random") for k in FULL_K for m in MODES]
    rows += [dict(experiment="sink", mode="mlp", k=0, ratio=r, policy="topk") for r in MLP_RATIOS]
    rows += [dict(experiment="sink", mode="mlp", k=0, ratio=r, policy=p)
             for r in MLP_HIGH for p in ("bottomk", "random", "lownorm")]
    for row in rows:
        row["name"] = f"{row['mode']}_k{row['k']}_{row['policy']}_{round(100 * row['ratio'])}"
    return rows


@contextmanager
def capture_block_output(block, width):
    """Keep the pre-norm output of one block when it runs at the given width."""
    captured = []
    handle = block.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach()) if output.shape[-1] == width else None)
    try:
        yield captured
    finally:
        handle.remove()


def wide_input(model, images, narrow_features):
    x = model.patch_embed(images)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)
    return x + model.alpha[0] * model.proj_layer(narrow_features)


def wide_blocks(model, x, p, mask=None, mode="freeze", k=0):
    """Run the 6H blocks; from block k on, freeze or drop tokens outside the mask."""
    for index, block in enumerate(model.blocks):
        if mask is not None and index == k and mode == "drop":
            keep = mask.nonzero()[:, 1].view(len(x), -1)  # equal count per row, CLS stays first
            x = x.gather(1, keep.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        candidate = block(x, p)
        frozen = mask is not None and mode == "freeze" and index >= k
        x = torch.where(mask.unsqueeze(-1), candidate, x) if frozen else candidate
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/imagenet1k")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/thinkingvit_3h6h_ema.pth")
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party/ThinkingViT")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Dataset ordering and random-policy seed")
    args = parser.parse_args()
    if not 1 <= args.samples <= 50000 or args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid samples, batch size, or workers")
    args.repo, args.data_dir, args.checkpoint = (args.repo.resolve(), args.data_dir.resolve(),
                                                  args.checkpoint.resolve())
    check_counts(args.data_dir / "validation", 50000)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    sys.path.insert(0, str(args.repo))
    global torch
    import torch
    import timm
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    from timm.data import create_transform, resolve_data_config
    from validate import _load_eval_checkpoint, RLDropping
    from selective_refinement import forward_narrow_with_attention, make_update_mask
    from mlp_selective_refinement import forward_mlp_selective_wide

    if not Path(timm.__file__).resolve().is_relative_to(args.repo):
        raise RuntimeError("Official custom timm is required")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    stages = (3, 6)
    model = timm.create_model("thinkingvit", pretrained=False, thinking_stages=stages)
    _load_eval_checkpoint(model, str(args.checkpoint), use_ema=True)
    RLDropping(model, used_heads=6, max_heads=6, ema=False)
    model = model.cuda().eval()
    if model.patch_embed.num_patches != 196 or model.num_prefix_tokens != 1:
        raise RuntimeError("This experiment expects 196 patches and one CLS token")
    config = resolve_data_config({}, model=model, use_test_size=True)
    dataset = ImageFolder(args.data_dir / "validation", transform=create_transform(**config, is_training=False))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.samples].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, shuffle=False)
    configs = experiment_configs()
    correct = {item["name"]: 0 for item in configs}
    # Same seed for every random config: freeze, drop and full-k see the same ranking per image.
    generators = {item["name"]: torch.Generator().manual_seed(args.seed)
                  for item in configs if item["policy"] == "random"}
    output = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-h1-diagnosis")
    output.mkdir(parents=True)
    print(f"Results: {output}", flush=True)
    source_files = [Path(__file__)] + [Path(__file__).parent / f"{name}.py"
                                       for name in ("selective_refinement", "mlp_selective_refinement")]
    environment = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "precision": "float32, AMP off, TF32 allowed (official default)", "preprocessing": config,
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
    }
    (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    narrow_p, wide_p = model._stage_ratio(stages[0]), model._stage_ratio(stages[1])
    narrow_width = int(narrow_p * model.embed_dim)
    per_image = {key: [] for key in ("entropy", "narrow_correct", "wide_correct")}
    sink = {key: [] for key in ("top_attention", "top_norm3", "top_norm6", "attention_top2_mass",
                                "norm3_at_top_attention", "norm3_max", "norm6_at_top_attention", "norm6_max",
                                "top2_overlap")}
    verified = False
    seen = 0
    with torch.no_grad():
        for batch_index, (images, targets) in enumerate(loader, 1):
            images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
            with capture_block_output(model.blocks[-1], narrow_width) as narrow_out:
                narrow_features, narrow_logits, attention = forward_narrow_with_attention(model, images, stages)
            narrow_norms = narrow_out[0].norm(dim=-1)
            entropy = model.entropy(narrow_logits)[:, 0]
            routed = ~(entropy < args.threshold)
            model.p = wide_p
            x0 = wide_input(model, images, narrow_features)
            wide_pre = wide_blocks(model, x0, wide_p)
            wide_logits = model.forward_head(model.norm(wide_pre, int(wide_p * model.embed_dim)))
            if not verified:
                reference = model.forward_head(model.forward_features(images, wide_p, narrow_features, stages))
                torch.testing.assert_close(wide_logits, reference, rtol=1e-5, atol=1e-6)
                full = torch.ones(x0.shape[:2], dtype=torch.bool, device=x0.device)
                for mode in MODES:
                    torch.testing.assert_close(wide_blocks(model, x0, wide_p, full, mode), wide_pre)
                verified = True
            per_image["entropy"].append(entropy.cpu())
            per_image["narrow_correct"].append((narrow_logits.argmax(1) == targets).cpu())
            per_image["wide_correct"].append((wide_logits.argmax(1) == targets).cpu())

            selected = routed.nonzero(as_tuple=True)[0]
            seen += len(images)
            if selected.numel():
                images_r, targets_r = images[selected], targets[selected]
                features_r, attention_r, x0_r = narrow_features[selected], attention[selected], x0[selected]
                norms3, norms6 = narrow_norms[selected, 1:], wide_pre[selected, 1:].norm(dim=-1)
                patch_attention = attention_r[:, 1:]
                top_attention = patch_attention.argmax(1, keepdim=True)
                top2_attention = patch_attention.topk(2, dim=1)
                top2_norm = norms3.topk(2, dim=1).indices
                for key, value in (
                        ("top_attention", top_attention[:, 0]), ("top_norm3", norms3.argmax(1)),
                        ("top_norm6", norms6.argmax(1)), ("attention_top2_mass", top2_attention.values.sum(1)),
                        ("norm3_at_top_attention", norms3.gather(1, top_attention)[:, 0] / norms3.median(1).values),
                        ("norm3_max", norms3.max(1).values / norms3.median(1).values),
                        ("norm6_at_top_attention", norms6.gather(1, top_attention)[:, 0] / norms6.median(1).values),
                        ("norm6_max", norms6.max(1).values / norms6.median(1).values),
                        ("top2_overlap", (top2_attention.indices[:, :, None] == top2_norm[:, None]).any(2).sum(1))):
                    sink[key].append(value.cpu())
                # lownorm keeps the lowest-norm patches, i.e. skips the highest-norm ones.
                lownorm_scores = torch.cat([attention_r[:, :1], -norms3], dim=1)
                for item in configs:
                    name = item["name"]
                    lownorm = item["policy"] == "lownorm"
                    mask = make_update_mask(lownorm_scores if lownorm else attention_r, item["ratio"],
                                            "topk" if lownorm else item["policy"], generators.get(name))
                    if item["mode"] == "mlp":
                        features = forward_mlp_selective_wide(model, images_r, features_r, mask, stages)
                    else:
                        model.p = wide_p
                        features = model.norm(wide_blocks(model, x0_r, wide_p, mask, item["mode"], item["k"]),
                                              int(wide_p * model.embed_dim))
                    correct[name] += int((model.forward_head(features).argmax(1) == targets_r).sum().item())
            if batch_index % 50 == 0 or seen == args.samples:
                print(f"Evaluated {seen}/{args.samples}", flush=True)
    if seen != args.samples or not verified:
        raise RuntimeError(f"Incomplete experiment: {seen}/{args.samples}")

    per_image = {key: torch.cat(value) for key, value in per_image.items()}
    sink = {key: torch.cat(value) for key, value in sink.items()}
    torch.save({**per_image, "sink": sink}, output / "per_image.pt")
    routed = ~(per_image["entropy"] < args.threshold)
    easy_correct = int(per_image["narrow_correct"][~routed].sum())
    routed_count = int(routed.sum())
    rows = [{**item, "routed_images": routed_count,
             "routed_top1": 100 * correct[item["name"]] / routed_count,
             "overall_top1": 100 * (easy_correct + correct[item["name"]]) / args.samples} for item in configs]
    report = {
        "complete": True, "samples": args.samples, "threshold": args.threshold,
        "routed_images": routed_count,
        "narrow_only_top1": 100 * float(per_image["narrow_correct"].float().mean()),
        "wide_all_top1": 100 * float(per_image["wide_correct"].float().mean()),
        "threshold_top1": 100 * (easy_correct + int(per_image["wide_correct"][routed].sum())) / args.samples,
        "routed_narrow_top1": 100 * float(per_image["narrow_correct"][routed].float().mean()),
        "routed_wide_top1": 100 * float(per_image["wide_correct"][routed].float().mean()),
        "full_path_matches_official": True,
        "semantics": {
            "freeze": "Unselected tokens keep their state from block k and stay as K/V (dense-mask oracle).",
            "drop": "Unselected tokens are removed from the 6H sequence before block k.",
            "mlp": "Attention on all tokens, MLP residual only on selected tokens (dense-mask oracle).",
            "selection": "3H final-block CLS attention; lownorm ranks by negative 3H final-block token norm.",
        },
        "results": rows,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
