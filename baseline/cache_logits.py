#!/usr/bin/env python3
"""Per-image logits for every execution option, so routing and output-mix rules can be studied on CPU (router_study.py).

Every image runs 3H and 6H at each keep ratio (default: eighths of the 196 patches, a uniform measurement grid that
contains the team's 25/50/75%) with the same Top-K token drop as evaluate_drop.py.
Saved: 3H logits and one 6H logits array per ratio (float32), 3H CLS feature and 3H final-block CLS attention (float16), the official
entropy (float32), targets and relative image paths. Images are read in ImageFolder order, so row i is dataset index i
(the old adaptive-drop cache is in seed-42 order and carries image_id; sort it by image_id to compare)."""
import argparse
from pathlib import Path
import torch
from evaluate_drop import STAGES, WIDE_P, add_paths, load_model, wide_logits
from evaluate_h1_diagnosis import wide_input
from selective_refinement import forward_narrow_with_attention, make_update_mask


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    add_paths(ap)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--n", type=int, required=True, help="expected image count (50000 for ImageNet val, 10000 for ImageNet-V2)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, help="smoke test: stop after this many batches and skip the count check")
    ap.add_argument("--ratios", type=float, nargs="+", default=[i / 8 for i in range(1, 9)])
    args = ap.parse_args()
    RATIOS = args.ratios
    assert RATIOS == sorted(RATIOS) and RATIOS[-1] == 1, RATIOS
    if args.output.exists():
        raise SystemExit(f"{args.output} exists")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_model(args.repo.resolve(), args.checkpoint.resolve(), args.weights)
    from timm.data import create_transform, resolve_data_config
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    cfg = resolve_data_config({}, model=model, use_test_size=True)
    root = args.data_dir / "validation"  # not resolved: the ImageNet-V2 folder is made of symlinks
    ds = ImageFolder(root, transform=create_transform(**cfg, is_training=False))
    assert ds.classes == [f"{i:04d}" for i in range(1000)] and (args.limit or len(ds) == args.n), (len(ds), ds.classes[:3])
    names = [f"wide_{round(1000 * r)}_logits" for r in RATIOS]  # per-mille, so 12.5% is wide_125
    out = {k: [] for k in ["narrow_logits", "narrow_cls", "attention", "entropy"] + names}
    with torch.no_grad():
        for bi, (images, _) in enumerate(DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True), 1):
            images = images.cuda(non_blocking=True)
            features, logits, attention = forward_narrow_with_attention(model, images, STAGES)
            x0 = wide_input(model, images, features)
            out["narrow_logits"].append(logits.float().cpu())  # float32: float16 flipped the argmax of about 30 near-tie images per option
            out["narrow_cls"].append(features[:, 0].half().cpu())
            out["attention"].append(attention.half().cpu())
            out["entropy"].append(model.entropy(logits)[:, 0].float().cpu())
            for r, name in zip(RATIOS, names):
                wide = wide_logits(model, x0, None if r == 1 else make_update_mask(attention, r, "topk"))
                if bi == 1 and r == 1:  # the full path must equal the official two-stage forward (same check as evaluate_drop.py)
                    model.p = WIDE_P
                    reference = model.forward_head(model.forward_features(images, WIDE_P, features, STAGES))
                    torch.testing.assert_close(wide, reference, rtol=1e-3, atol=1e-3)
                out[name].append(wide.float().cpu())
            if bi % 100 == 0:
                print(f"cached {bi * args.batch_size}/{len(ds)}", flush=True)
            if args.limit and bi >= args.limit:
                break
    data = {k: torch.cat(v) for k, v in out.items()}
    n = len(data["entropy"])
    data["ratios"] = torch.tensor(RATIOS, dtype=torch.float64)
    data["target"] = torch.tensor(ds.targets[:n], dtype=torch.int64)
    data["image_path"] = [str(Path(p).relative_to(root)) for p, _ in ds.samples[:n]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.output)
    for k in ["narrow_logits"] + names:
        print(k, f"top1 {100 * (data[k].float().argmax(1) == data['target']).float().mean().item():.3f}%", flush=True)
    print("CACHE_DONE", args.output, n, flush=True)


if __name__ == "__main__":
    main()
