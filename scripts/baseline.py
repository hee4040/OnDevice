#!/usr/bin/env python3
"""Run unmodified official accuracy evaluation or measure its batch-1 forward."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

from prepare_imagenet import check_counts

ROOT = Path(__file__).resolve().parents[1]


def parse_log(text):
    matches = re.findall(r"--result\s*(\{.*?\})", text, re.DOTALL)
    if not matches:
        raise ValueError("No completed official --result block found")
    result = json.loads(matches[-1])
    dispatch = re.findall(r"Model\s+\d+\s+Dispatched:\s*[\d.]+%\s+\((\d+)/(\d+)\)", text)
    if len(dispatch) != 2 or any(int(total) != 50000 for _, total in dispatch):
        raise ValueError("Expected two dispatch counts for the full 50,000-image validation set")
    counts = [int(count) for count, _ in dispatch]
    if sum(counts) != 50000:
        raise ValueError("Invalid dispatch count total")
    result["stage_2_fraction"] = counts[1] / 50000
    result["gmacs_official_routing_estimate"] = 1.25 + 4.6 * result["stage_2_fraction"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["accuracy", "latency"], default="accuracy")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/imagenet1k")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/thinkingvit_3h6h_ema.pth")
    p.add_argument("--repo", type=Path, default=ROOT / "third_party/ThinkingViT")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--thresholds", nargs="+", type=float, default=[10.0, 0.0, 1.0])
    p.add_argument("--samples", type=int, default=1000, help="Latency samples; accuracy always uses all 50,000")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.batch_size < 1 or not 1 <= args.samples <= 50000 or args.warmup < 1:
        p.error("Invalid batch-size, samples, or warmup")
    args.repo = args.repo.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.data_dir = args.data_dir.resolve()
    for path in [args.repo / "validate.py", args.checkpoint, args.data_dir / "validation"]:
        if not path.exists():
            p.error(f"Missing {path}; complete setup and dataset preparation first")
    check_counts(args.data_dir / "validation", 50000)
    sys.path.insert(0, str(args.repo))
    import torch
    import timm
    if not Path(timm.__file__).resolve().is_relative_to(args.repo):
        raise RuntimeError("Incorrect timm import: official custom timm is required")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; GPU measurements cannot run in this environment.")
    os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # Matches official validate.py.
    torch.backends.cudnn.benchmark = True
    output = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + args.mode)
    output.mkdir(parents=True)
    metadata = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                "precision": "float32, AMP off, TF32 allowed (official default)",
                "repo_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.repo, text=True).strip(),
                "repo_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=args.repo, text=True),
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "warning": "Inspected upstream forward computes both stages before selecting logits. Routing GMACs are estimates, not executed operation counts."}
    (output / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    if args.mode == "accuracy":
        for threshold in args.thresholds:
            command = [sys.executable, "-u", "validate.py", "--model", "thinkingvit",
                       "--checkpoint", str(args.checkpoint), "--use-ema", "--thinking_stages", "3", "6",
                       "--data-dir", str(args.data_dir), "--split", "validation",
                       "--batch-size", str(args.batch_size), "--threshold", str(threshold)]
            (output / f"threshold_{threshold}.command.json").write_text(json.dumps(command, indent=2) + "\n")
            log_path = output / f"threshold_{threshold}.log"
            with log_path.open("w") as log:
                subprocess.run(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, check=True)
            rows.append({"threshold": threshold, **parse_log(log_path.read_text())})
            (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(rows[-1], flush=True)
    else:
        from validate import _load_eval_checkpoint, RLDropping
        from timm.data import create_transform, resolve_data_config
        from torchvision.datasets import ImageFolder
        model = timm.create_model("thinkingvit", pretrained=False, thinking_stages=[3, 6])
        _load_eval_checkpoint(model, str(args.checkpoint), use_ema=True)
        RLDropping(model, used_heads=6, max_heads=6, ema=False)
        model = model.cuda().eval()
        config = resolve_data_config({}, model=model, use_test_size=True)
        (output / "preprocessing.json").write_text(json.dumps(config, indent=2) + "\n")
        dataset = ImageFolder(args.data_dir / "validation", transform=create_transform(**config, is_training=False))
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))[:args.samples].tolist()
        (output / "sample_indices.json").write_text(json.dumps(indices) + "\n")
        with torch.no_grad():
            for threshold in args.thresholds:
                x = dataset[indices[0]][0].unsqueeze(0).cuda()
                for _ in range(args.warmup):
                    model(x, threshold=threshold, thinking_stages=[3, 6])
                del x
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                times, dispatch = [], []
                for index in indices:
                    x = dataset[index][0].unsqueeze(0).cuda()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    logits, stage = model(x, threshold=threshold, thinking_stages=[3, 6])
                    torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
                    dispatch.append(int(stage.item()))
                    del x, logits, stage
                milliseconds = torch.tensor(times, dtype=torch.float64)
                rows.append({"threshold": threshold, "samples": len(times), "batch_size": 1,
                             "mean_ms": milliseconds.mean().item(), "p50_ms": milliseconds.quantile(.5).item(),
                             "p95_ms": milliseconds.quantile(.95).item(),
                             "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                             "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                             "stage_2_fraction": sum(dispatch) / len(dispatch),
                             "measurement": "official forward + dispatch; synchronized wall time; excludes preprocessing and H2D; memory includes model"})
                (output / f"threshold_{threshold}.timings.json").write_text(json.dumps({"ms": times, "stages": dispatch}) + "\n")
                (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
                print(rows[-1], flush=True)
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
