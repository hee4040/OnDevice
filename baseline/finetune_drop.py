#!/usr/bin/env python3
"""Fine-tune ThinkingViT so the drop Top-K 6H pass imitates the full 6H pass.

Student: 3H, then 6H on CLS + the Top-K patches ranked by its own 3H CLS attention
(ratio drawn per batch from --ratios); the other patches leave the 6H sequence.
Teacher: the frozen official model, 3H and full 6H on the same crop.
Loss = 0.5 KL(3H) + 0.5 KL(drop 6H): the official 0.5/0.5 stage weighting with the
teacher's answers instead of labels. 3H shares the weights, so its term keeps 3H in place.
Trains on ImageNet-1K train only; validation stays for evaluate_drop.py.
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

import torch
import torch.nn.functional as F

from evaluate_drop import ROOT, STAGES, NARROW_P, WIDE_P, add_paths, load_model, wide_logits
from evaluate_h1_diagnosis import wide_input
from selective_refinement import forward_narrow_with_attention, make_update_mask


def distill(student, teacher):
    return F.kl_div(F.log_softmax(student.float(), 1), F.log_softmax(teacher.float(), 1),
                    reduction="batchmean", log_target=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_paths(parser)
    parser.add_argument("--ratios", nargs="+", type=float, default=[0.25, 0.5])
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, help="Stop early (smoke test)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path, help="student.pth of an interrupted run (saved every 2000 steps)")
    parser.add_argument("--start-step", type=int, default=0, help="Step of --resume; the schedule continues from it")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # eval() on purpose: this model has no dropout or drop-path, and the attention capture requires it.
    student = load_model(args.repo.resolve(), args.checkpoint.resolve())
    teacher = copy.deepcopy(student).requires_grad_(False)
    if args.resume:  # optimizer moments restart; the learning-rate schedule does not
        student.load_state_dict(torch.load(args.resume, map_location="cuda", weights_only=True))
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from timm.data import create_transform, resolve_data_config
    config = resolve_data_config({}, model=student)
    dataset = ImageFolder(args.data_dir / "train", transform=create_transform(**config, is_training=True))
    if len(dataset) != 1281167:
        raise RuntimeError(f"Expected 1,281,167 train images, found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4,
                        generator=torch.Generator().manual_seed(args.seed + args.start_step))
    total = math.ceil(args.epochs * len(loader))
    if args.max_steps:
        total = min(total, args.max_steps)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1, (s + args.start_step + 1) / args.warmup)
                                                 * 0.5 * (1 + math.cos(math.pi * min(s + args.start_step, total) / total)))
    output = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-finetune-drop")
    output.mkdir(parents=True)
    (output / "arguments.json").write_text(json.dumps(
        {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "steps": total,
         "train_transform": str(dataset.transform), "gpu": torch.cuda.get_device_name()}, indent=2) + "\n")
    print(f"Results: {output}  steps: {total}", flush=True)

    step, start, log = args.start_step, time.time(), []
    if args.resume:
        log = [e for e in json.loads((args.resume.parent / "train_log.json").read_text()) if e["step"] <= step]
    while step < total:
        for images, _ in loader:
            images = images.cuda(non_blocking=True)
            ratio = random.choice(args.ratios)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    teacher.p = NARROW_P
                    teacher_features = teacher.forward_features(images, NARROW_P)
                    teacher_narrow = teacher.forward_head(teacher_features)
                    teacher_wide = wide_logits(teacher, wide_input(teacher, images, teacher_features))
                features, narrow, attention = forward_narrow_with_attention(student, images, STAGES)
                wide = wide_logits(student, wide_input(student, images, features),
                                   make_update_mask(attention.float(), ratio, "topk"))
            loss_narrow, loss_wide = distill(narrow, teacher_narrow), distill(wide, teacher_wide)
            loss = 0.5 * loss_narrow + 0.5 * loss_wide
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            if step % 100 == 0 or step == total:
                log.append({"step": step, "loss_3h": loss_narrow.item(), "loss_drop_6h": loss_wide.item(),
                            "ratio": ratio, "lr": schedule.get_last_lr()[0],
                            "images_per_s": (step - args.start_step) * args.batch_size / (time.time() - start)})
                print(json.dumps(log[-1]), flush=True)
                (output / "train_log.json").write_text(json.dumps(log) + "\n")
            if step % 2000 == 0 or step == total:  # survive a reboot of the shared PC
                torch.save(student.state_dict(), output / "student.pth")
            if step == total:
                break
    print(f"Done in {(time.time() - start) / 60:.1f} min: {output / 'student.pth'}", flush=True)


if __name__ == "__main__":
    main()
