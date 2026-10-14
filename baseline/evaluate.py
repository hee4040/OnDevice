#!/usr/bin/env python3
"""Verify ThinkingViT early exit; measure batch-1 latency and batched accuracy."""
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
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_imagenet import check_counts


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["latency", "verify", "accuracy"], default="latency")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/imagenet1k")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/thinkingvit_3h6h_ema.pth")
    parser.add_argument("--repo", type=Path, default=ROOT / "third_party/ThinkingViT")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[10.0, 0.0, 1.0])
    parser.add_argument("--samples", type=int, default=1000, help="Latency/verify samples; accuracy uses all 50,000")
    parser.add_argument("--verify-samples", type=int, default=32, help="Preflight comparisons before latency/accuracy")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128, help="Accuracy/batched verification only; latency stays batch-1")
    parser.add_argument("--workers", type=int, default=4, help="Accuracy/verification data loader workers")
    parser.add_argument("--compare-official", action="store_true", help="Compare every accuracy sample with official forward in the same batch")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.compare_official and args.mode != "accuracy":
        parser.error("--compare-official requires --mode accuracy")
    if (not 1 <= args.samples <= 50000 or not 1 <= args.verify_samples <= 50000 or args.warmup < 1
            or args.batch_size < 1 or args.workers < 0):
        parser.error("Invalid samples, verify-samples, or warmup")
    args.repo = args.repo.resolve()
    args.data_dir = args.data_dir.resolve()
    args.checkpoint = args.checkpoint.resolve()
    check_counts(args.data_dir / "validation", 50000)
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    sys.path.insert(0, str(args.repo))
    import torch
    import timm
    from torchvision.datasets import ImageFolder
    from timm.data import create_transform, resolve_data_config
    from validate import _load_eval_checkpoint, RLDropping
    from early_exit import forward_early_exit, forward_early_exit_batch
    from torch.utils.data import DataLoader, Subset

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
    count = 50000 if args.mode == "accuracy" else args.samples
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:count].tolist()
    output = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-early-exit-" + args.mode)
    output.mkdir(parents=True)
    print(f"Results: {output}", flush=True)
    save_json(output / "sample_indices.json", indices)
    save_json(output / "preprocessing.json", config)
    save_json(output / "environment.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "precision": "float32, AMP off, TF32 allowed (official default)",
        "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
        "repo_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "baseline_source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                   for name in ("early_exit.py", "evaluate.py")},
    })

    def official(x, threshold):
        return model(x, threshold=threshold, thinking_stages=[3, 6])

    def early(x, threshold):
        return forward_early_exit(model, x, threshold)

    def loader(sample_indices):
        return DataLoader(Subset(dataset, sample_indices), batch_size=args.batch_size,
                          num_workers=args.workers, pin_memory=True, shuffle=False)

    feature_calls = 0
    original_features = model.forward_features

    def counted_features(*positional, **keywords):
        nonlocal feature_calls
        feature_calls += 1
        return original_features(*positional, **keywords)

    rows, verification = [], []
    with torch.no_grad():
        for threshold in args.thresholds:
            verify_indices = indices if args.mode == "verify" else indices[:args.verify_samples]
            max_error, wide_calls = 0.0, 0
            for n, index in enumerate(verify_indices, 1):
                x = dataset[index][0].unsqueeze(0).cuda()
                expected, expected_stage = official(x, threshold)
                feature_calls = 0
                with patch.object(model, "forward_features", new=counted_features):
                    actual, stage = early(x, threshold)
                    assert feature_calls == 1 + int(stage.item()), "Stage execution count mismatch"
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(stage, expected_stage, rtol=0, atol=0)
                assert torch.equal(actual.argmax(1), expected.argmax(1)), "Top-1 mismatch"
                max_error = max(max_error, (actual - expected).abs().max().item())
                wide_calls += int(stage.item())
                del x, expected, expected_stage, actual, stage
                if n % 100 == 0:
                    print(f"verify threshold={threshold}: {n}/{len(verify_indices)}", flush=True)
            verification.append({"threshold": threshold, "samples": len(verify_indices),
                                 "max_abs_logit_error": max_error, "stage_2_calls": wide_calls,
                                 "logits_close": True, "top1_and_routing_equal": True,
                                 "execution_counts_verified": True})
            save_json(output / "verification.json", verification)
            print(f"Verified: {verification[-1]}", flush=True)
            if args.mode in {"accuracy", "verify"}:
                batch_error, batch_wide = 0.0, 0
                for images, _ in loader(verify_indices):
                    x = images.cuda(non_blocking=True)
                    expected, expected_stage = official(x, threshold)
                    sizes = []

                    def counted_batch_features(*positional, **keywords):
                        sizes.append(positional[0].shape[0])
                        return original_features(*positional, **keywords)

                    with patch.object(model, "forward_features", new=counted_batch_features):
                        actual, stage = forward_early_exit_batch(model, x, threshold)
                    selected_count = int(stage.sum().item())
                    assert sizes == [len(x)] + ([selected_count] if selected_count else []), "Incorrect wide subset"
                    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
                    torch.testing.assert_close(stage, expected_stage, rtol=0, atol=0)
                    assert torch.equal(actual.argmax(1), expected.argmax(1)), "Batched Top-1 mismatch"
                    batch_error = max(batch_error, (actual - expected).abs().max().item())
                    batch_wide += selected_count
                    del x, expected, expected_stage, actual, stage
                verification[-1]["batched"] = {"batch_size": args.batch_size, "samples": len(verify_indices),
                                               "max_abs_logit_error": batch_error, "stage_2_images": batch_wide,
                                               "logits_close": True, "top1_and_routing_equal": True,
                                               "execution_sizes_verified": True}
                save_json(output / "verification.json", verification)
                print(f"Batched verification: {verification[-1]['batched']}", flush=True)
            if args.mode == "verify":
                continue
            if args.mode == "accuracy":
                correct1, correct5, stage2 = 0, 0, 0
                reference1, reference5, prediction_differences, routing_differences = 0, 0, 0, 0
                reference_error = 0.0
                seen = 0
                for batch_index, (images, targets) in enumerate(loader(indices), 1):
                    x = images.cuda(non_blocking=True)
                    logits, stage = forward_early_exit_batch(model, x, threshold)
                    targets = targets.cuda(non_blocking=True)
                    predictions = logits.topk(5, dim=1).indices
                    correct1 += int((predictions[:, 0] == targets).sum().item())
                    correct5 += int((predictions == targets[:, None]).any(dim=1).sum().item())
                    stage2 += int(stage.sum().item())
                    if args.compare_official:
                        reference, reference_stage = official(x, threshold)
                        reference_predictions = reference.topk(5, dim=1).indices
                        reference1 += int((reference_predictions[:, 0] == targets).sum().item())
                        reference5 += int((reference_predictions == targets[:, None]).any(dim=1).sum().item())
                        prediction_differences += int((predictions[:, 0] != reference_predictions[:, 0]).sum().item())
                        routing_differences += int((stage != reference_stage).sum().item())
                        reference_error = max(reference_error, (reference - logits).abs().max().item())
                        del reference, reference_stage, reference_predictions
                    seen += len(images)
                    del x, logits, stage, predictions
                    if batch_index % 10 == 0 or seen == count:
                        print(f"accuracy threshold={threshold}: {seen}/{count}, top1={100 * correct1 / seen:.3f}", flush=True)
                assert seen == 50000, "Incomplete validation set"
                rows.append({"implementation": "early_exit", "threshold": threshold, "samples": count,
                             "batch_size": args.batch_size, "top1": 100 * correct1 / count, "top5": 100 * correct5 / count,
                             "stage_2_fraction": stage2 / count, "stage_2_execution_fraction": stage2 / count})
                if args.compare_official:
                    rows[-1].update({"official_same_batch_top1": 100 * reference1 / count,
                                     "official_same_batch_top5": 100 * reference5 / count,
                                     "top1_prediction_disagreements": prediction_differences,
                                     "routing_disagreements": routing_differences,
                                     "max_abs_logit_error": reference_error})
                print(rows[-1], flush=True)
            else:
                for name, forward in (("official", official), ("early_exit", early)):
                    x = dataset[indices[0]][0].unsqueeze(0).cuda()
                    # Warm both branches regardless of the first image's entropy.
                    for _ in range(args.warmup):
                        forward(x, 10.0)
                        forward(x, 0.0)
                    del x
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    times, stages = [], []
                    for n, index in enumerate(indices, 1):
                        x = dataset[index][0].unsqueeze(0).cuda()
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        logits, stage = forward(x, threshold)
                        torch.cuda.synchronize()
                        times.append((time.perf_counter() - start) * 1000)
                        stages.append(int(stage.item()))
                        del x, logits, stage
                        if n % 100 == 0:
                            print(f"{name} threshold={threshold}: {n}/{count}", flush=True)
                    ms = torch.tensor(times, dtype=torch.float64)
                    rows.append({"implementation": name, "threshold": threshold, "samples": count,
                                 "batch_size": 1, "mean_ms": ms.mean().item(), "p50_ms": ms.quantile(.5).item(),
                                 "p95_ms": ms.quantile(.95).item(),
                                 "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                                 "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                                 "stage_2_fraction": sum(stages) / count,
                                 "stage_2_execution_fraction": 1.0 if name == "official" else sum(stages) / count,
                                 "measurement": "batch-1 synchronized wall time; includes gate synchronization; "
                                                "excludes preprocessing and H2D; memory includes model"})
                    save_json(output / f"{name}_{threshold}.timings.json", {"ms": times, "stages": stages})
                    save_json(output / "summary.json", rows)
                    print(rows[-1], flush=True)
            save_json(output / "summary.json", rows)
        if rows:
            with (output / "summary.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
