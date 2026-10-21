#!/usr/bin/env python3
"""Evaluate H1 with a dense-mask selective 6H accuracy oracle."""
import argparse
import csv
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


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def experiment_configs(ratios, policies, random_seeds):
    configs = []
    for ratio in ratios:
        if ratio in {0.0, 1.0}:
            configs.append({"ratio": ratio, "policy": "topk", "random_seed": None})
            continue
        for policy in policies:
            seeds = random_seeds if policy == "random" else [None]
            for seed in seeds:
                configs.append({"ratio": ratio, "policy": policy, "random_seed": seed})
    return configs


def config_name(config):
    percent = round(100 * config["ratio"])
    suffix = f"_seed{config['random_seed']}" if config["random_seed"] is not None else ""
    return f"{config['policy']}_{percent}{suffix}"


def empty_metrics(config):
    return {
        **config,
        "name": config_name(config),
        "samples": 0,
        "routed_images": 0,
        "top1_correct": 0,
        "top5_correct": 0,
        "routed_top1_correct": 0,
        "baseline_corrected_images": 0,
        "corrections_retained": 0,
        "baseline_wide_correct_images": 0,
        "wide_correct_retained": 0,
        "baseline_regressed_images": 0,
        "regressions_recovered": 0,
        "top1_disagreements_with_full_wide_on_routed": 0,
    }


def finalize(metrics, patch_tokens):
    result = dict(metrics)
    samples, routed = result["samples"], result["routed_images"]
    corrected = result["baseline_corrected_images"]
    wide_correct = result["baseline_wide_correct_images"]
    regressed = result["baseline_regressed_images"]
    result.update({
        "selected_patch_tokens": round(patch_tokens * result["ratio"]),
        "updated_tokens_including_cls": 1 + round(patch_tokens * result["ratio"]),
        "overall_top1": 100 * result["top1_correct"] / samples,
        "overall_top5": 100 * result["top5_correct"] / samples,
        "routed_top1": 100 * result["routed_top1_correct"] / routed if routed else None,
        "correction_retention": result["corrections_retained"] / corrected if corrected else None,
        "wide_correct_retention": result["wide_correct_retained"] / wide_correct if wide_correct else None,
        "regression_recovery": result["regressions_recovered"] / regressed if regressed else None,
        "implementation": "dense-mask accuracy oracle; no FLOPs or latency reduction",
    })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/imagenet1k")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/thinkingvit_3h6h_ema.pth")
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party/ThinkingViT")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0, .25, .5, .75, 1])
    parser.add_argument("--policies", nargs="+", choices=["topk", "bottomk", "random"],
                        default=["topk", "bottomk", "random"])
    parser.add_argument("--random-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Dataset ordering seed")
    args = parser.parse_args()
    if (not 1 <= args.samples <= 50000 or args.batch_size < 1 or args.workers < 0
            or not args.random_seeds or any(not 0 <= ratio <= 1 for ratio in args.ratios)):
        parser.error("Invalid samples, batch size, workers, ratios, or random seeds")
    args.ratios = sorted(set(args.ratios))
    args.repo, args.data_dir, args.checkpoint = (args.repo.resolve(), args.data_dir.resolve(),
                                                  args.checkpoint.resolve())
    check_counts(args.data_dir / "validation", 50000)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    sys.path.insert(0, str(args.repo))
    import torch
    import timm
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    from timm.data import create_transform, resolve_data_config
    from validate import _load_eval_checkpoint, RLDropping
    from selective_refinement import forward_narrow_with_attention, forward_selective_wide, selective_logits

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
    config = resolve_data_config({}, model=model, use_test_size=True)
    dataset = ImageFolder(args.data_dir / "validation", transform=create_transform(**config, is_training=False))
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.samples].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=True, shuffle=False)
    configs = experiment_configs(args.ratios, args.policies, args.random_seeds)
    metrics = {config_name(item): empty_metrics(item) for item in configs}
    generators = {
        # Separate generators with the same seed make each ratio use prefixes of
        # one per-image random ranking, which gives a controlled ratio sweep.
        config_name(item): torch.Generator().manual_seed(args.seed + 1009 * item["random_seed"])
        for item in configs if item["policy"] == "random"
    }
    output = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                                  + "-selective-refinement")
    output.mkdir(parents=True)
    print(f"Results: {output}", flush=True)
    save_json(output / "sample_indices.json", indices)
    source_files = (Path(__file__), Path(__file__).parent / "selective_refinement.py")
    save_json(output / "environment.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "precision": "float32, AMP off, TF32 allowed (official default)", "preprocessing": config,
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "repo_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
        "semantics": "Static per-image token mask across all 12 wide blocks; CLS always updated; "
                     "selected queries use all current tokens as K/V; dense compute then masked state update.",
        "claim_scope": "Accuracy/H1 oracle only. Dense block compute does not reduce FLOPs or latency.",
    })
    patch_tokens = model.patch_embed.num_patches
    if patch_tokens != 196 or model.num_prefix_tokens != 1:
        raise RuntimeError("This experiment expects 196 patches and one CLS token")
    full_mask_error = 0.0
    full_mask_verified = False
    seen = 0
    with torch.no_grad():
        for batch_index, (images, targets_cpu) in enumerate(loader, 1):
            images = images.cuda(non_blocking=True)
            targets = targets_cpu.cuda(non_blocking=True)
            narrow_features, narrow_logits, attention = forward_narrow_with_attention(model, images, stages)
            routed = (~(model.entropy(narrow_logits) < args.threshold))[:, 0]
            selected = routed.nonzero(as_tuple=True)[0]
            narrow_top5 = narrow_logits.topk(5, dim=1).indices
            narrow_correct = narrow_top5[:, 0] == targets
            easy = ~routed
            full_logits = None
            if selected.numel():
                selected_images = images.index_select(0, selected)
                selected_features = narrow_features.index_select(0, selected)
                selected_attention = attention.index_select(0, selected)
                selected_targets = targets.index_select(0, selected)
                selected_narrow_correct = narrow_correct.index_select(0, selected)
                full_mask = torch.ones((len(selected), patch_tokens + 1), dtype=torch.bool, device=images.device)
                full_features = forward_selective_wide(model, selected_images, selected_features, full_mask, stages)
                full_logits = model.forward_head(full_features)
                full_top1 = full_logits.argmax(1)
                full_correct = full_top1 == selected_targets
                corrected = ~selected_narrow_correct & full_correct
                regressed = selected_narrow_correct & ~full_correct
                if not full_mask_verified:
                    p = model.p = model._stage_ratio(stages[1])
                    reference_features = model.forward_features(selected_images, p, selected_features, stages)
                    reference_logits = model.forward_head(reference_features)
                    torch.testing.assert_close(full_logits, reference_logits, rtol=1e-5, atol=1e-6)
                    if not torch.equal(full_top1, reference_logits.argmax(1)):
                        raise AssertionError("100% selective path Top-1 differs from official full-wide path")
                    full_mask_error = (full_logits - reference_logits).abs().max().item()
                    full_mask_verified = True
            for item in configs:
                name = config_name(item)
                row = metrics[name]
                row["samples"] += len(images)
                row["routed_images"] += int(routed.sum().item())
                row["top1_correct"] += int(narrow_correct[easy].sum().item())
                row["top5_correct"] += int((narrow_top5[easy] == targets[easy, None]).any(1).sum().item())
                if not selected.numel():
                    continue
                if item["ratio"] == 1.0:
                    logits = full_logits
                else:
                    logits, mask = selective_logits(
                        model, selected_images, selected_features, selected_attention, item["ratio"],
                        item["policy"], generators.get(name), stages,
                    )
                    expected_updates = 1 + round(patch_tokens * item["ratio"])
                    if not torch.equal(mask.sum(1), torch.full((len(selected),), expected_updates,
                                                               device=mask.device)):
                        raise AssertionError("Incorrect number of updated tokens")
                predictions = logits.topk(5, dim=1).indices
                actual_correct = predictions[:, 0] == selected_targets
                row["top1_correct"] += int(actual_correct.sum().item())
                row["top5_correct"] += int((predictions == selected_targets[:, None]).any(1).sum().item())
                row["routed_top1_correct"] += int(actual_correct.sum().item())
                row["baseline_corrected_images"] += int(corrected.sum().item())
                row["corrections_retained"] += int((corrected & actual_correct).sum().item())
                row["baseline_wide_correct_images"] += int(full_correct.sum().item())
                row["wide_correct_retained"] += int((full_correct & actual_correct).sum().item())
                row["baseline_regressed_images"] += int(regressed.sum().item())
                row["regressions_recovered"] += int((regressed & actual_correct).sum().item())
                row["top1_disagreements_with_full_wide_on_routed"] += int(
                    (predictions[:, 0] != full_top1).sum().item())
            seen += len(images)
            if batch_index % 10 == 0 or seen == args.samples:
                print(f"Evaluated {seen}/{args.samples}", flush=True)
            del images, targets, narrow_features, narrow_logits, attention
    if seen != args.samples:
        raise RuntimeError(f"Incomplete experiment: {seen}/{args.samples}")
    if not full_mask_verified:
        raise RuntimeError("No image was routed to 6H; full-mask equivalence was not verified")
    rows = [finalize(metrics[config_name(item)], patch_tokens) for item in configs]
    report = {
        "complete": True,
        "full_validation": args.samples == 50000,
        "samples": args.samples,
        "threshold": args.threshold,
        "full_mask_matches_official": True,
        "full_mask_max_abs_logit_error_first_batch": full_mask_error,
        "results": rows,
    }
    save_json(output / "summary.json", report)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
