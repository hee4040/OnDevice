#!/usr/bin/env python3
"""Choose PEFT learning rates using a held-out part of train, never validation."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from evaluate_drop import ROOT, NARROW_P, add_paths, load_model, wide_logits
from evaluate_h1_diagnosis import wide_input
from selective_refinement import forward_narrow_with_attention, make_update_mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    teacher = load_model(args.repo.resolve(), args.checkpoint.resolve())
    student = load_model(args.repo.resolve(), args.checkpoint.resolve(), weights=args.weights)
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader, Subset
    from timm.data import create_transform, resolve_data_config
    config = resolve_data_config({}, model=teacher, use_test_size=True)
    dataset = ImageFolder(args.data_dir / "train", transform=create_transform(**config, is_training=False))
    if len(dataset) != 1281167:
        raise ValueError("Calibration must come from the complete train split")
    indices = torch.load(args.indices, map_location="cpu", weights_only=True)
    loader = DataLoader(Subset(dataset, indices), batch_size=64, num_workers=4, pin_memory=True)
    count, hard_count, correct = 0, 0, 0
    kl_total, kl_hard = 0., 0.
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.cuda(), targets.cuda()
            teacher.p = NARROW_P
            features_t = teacher.forward_features(images, NARROW_P)
            narrow_t = teacher.forward_head(features_t)
            logits_t = wide_logits(teacher, wide_input(teacher, images, features_t))
            features, narrow, attention = forward_narrow_with_attention(student, images)
            torch.testing.assert_close(narrow, narrow_t, rtol=0, atol=0)
            logits = wide_logits(student, wide_input(student, images, features), make_update_mask(attention, .5))
            routed = student.entropy(narrow)[:, 0] >= 1.
            kl = F.kl_div(F.log_softmax(logits.float(), 1), F.log_softmax(logits_t.float(), 1),
                          log_target=True, reduction="none").sum(1)
            kl_total += kl.sum().item()
            kl_hard += kl[routed].sum().item()
            hard_count += routed.sum().item()
            correct += (torch.where(routed[:, None], logits, narrow).argmax(1) == targets).sum().item()
            count += len(images)
    if count != len(indices) or not hard_count:
        raise ValueError("Incomplete or empty routed calibration set")
    report = {"split": "train calibration", "samples": count, "routed": hard_count,
              "kd_all": kl_total / count, "kd_routed": kl_hard / hard_count,
              "top1": 100 * correct / count, "elapsed_seconds": time.perf_counter() - start,
              "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "indices_sha256": hashlib.sha256(args.indices.read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
