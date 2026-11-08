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
from drop_ft_losses import aligned_feature_loss, capture_wide_features


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
    parser.add_argument("--resume", type=Path, help="resume.pth with optimizer/RNG, or legacy student.pth")
    parser.add_argument("--start-step", type=int, default=0, help="Step of --resume; the schedule continues from it")
    parser.add_argument("--micro-batch-size", type=int, default=32,
                        help="Accumulate gradients to --batch-size without changing the effective batch")
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument("--full-weight", type=float, default=0.0)
    parser.add_argument("--feature-weight", type=float, default=0.0)
    parser.add_argument("--feature-layers", nargs="+", type=int, default=[3, 7, 11],
                        help="Zero-based wide block indices")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()
    if (args.batch_size < 1 or args.micro_batch_size < 1 or args.workers < 0 or args.warmup < 1
            or args.epochs <= 0 or args.save_every < 1 or args.lr <= 0
            or any(w < 0 for w in (args.ce_weight, args.full_weight, args.feature_weight))
            or any(not 0 <= layer < 12 for layer in args.feature_layers)
            or not args.ratios or any(not 0 <= ratio <= 1 for ratio in args.ratios)):
        parser.error("Invalid batch, workers, schedule, loss weights, feature layers, or token ratios")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # eval() on purpose: this model has no dropout or drop-path, and the attention capture requires it.
    student = load_model(args.repo.resolve(), args.checkpoint.resolve())
    teacher = copy.deepcopy(student).requires_grad_(False)
    resume_state = None
    if args.resume:
        saved = torch.load(args.resume, map_location="cuda", weights_only=True)
        if "student" in saved:
            resume_state = saved
            student.load_state_dict(saved["student"])
            args.start_step = int(saved["step"])
        else:  # Backward compatibility with the 4090 run's plain state dict.
            student.load_state_dict(saved)
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from timm.data import create_transform, resolve_data_config
    config = resolve_data_config({}, model=student)
    dataset = ImageFolder(args.data_dir / "train", transform=create_transform(**config, is_training=True))
    if len(dataset) != 1281167:
        raise RuntimeError(f"Expected 1,281,167 train images, found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True,
                        **({"persistent_workers": True, "prefetch_factor": 2} if args.workers else {}),
                        generator=torch.Generator().manual_seed(args.seed + args.start_step))
    total = math.ceil(args.epochs * len(loader))
    if args.max_steps:
        total = min(total, args.max_steps)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    schedule_offset = 0 if resume_state is not None else args.start_step
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1, (s + schedule_offset + 1) / args.warmup)
                                                 * 0.5 * (1 + math.cos(math.pi * min(s + schedule_offset, total) / total)))
    if resume_state is not None:
        previous = resume_state["arguments"]
        for key in ("epochs", "max_steps", "batch_size", "micro_batch_size", "lr", "warmup", "seed",
                    "ratios", "ce_weight", "full_weight", "feature_weight", "feature_layers"):
            if previous.get(key) != getattr(args, key):
                raise ValueError(f"Cannot resume with a changed {key}")
        optimizer.load_state_dict(resume_state["optimizer"])
        schedule.load_state_dict(resume_state["schedule"])
        random.setstate(resume_state["python_rng"])
        torch.set_rng_state(resume_state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([state.cpu() for state in resume_state["cuda_rng"]])
    output = args.output_dir or ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-finetune-drop")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "student.pth").exists() and not args.resume:
        raise ValueError("Output already contains weights; choose a new directory or resume")
    (output / "arguments.json").write_text(json.dumps(
        {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "steps": total,
         "train_transform": str(dataset.transform), "gpu": torch.cuda.get_device_name()}, indent=2) + "\n")
    print(f"Results: {output}  steps: {total}", flush=True)

    step, start, log = args.start_step, time.time(), []
    if args.resume:
        log_file = args.resume.parent / "train_log.json"
        if log_file.exists():
            log = [e for e in json.loads(log_file.read_text()) if e["step"] <= step]
    def save():
        # Atomic replacements keep the previous checkpoint usable if interrupted.
        torch.save(student.state_dict(), output / "student.tmp.pth")
        (output / "student.tmp.pth").replace(output / "student.pth")
        arguments = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "schedule": schedule.state_dict(), "step": step, "arguments": arguments,
                    "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all()}, output / "resume.tmp.pth")
        (output / "resume.tmp.pth").replace(output / "resume.pth")

    while step < total:
        epoch = step // len(loader)
        skip_batches = step % len(loader)
        loader.generator.manual_seed(args.seed + epoch)
        for batch_index, (images, targets) in enumerate(loader):
            if batch_index < skip_batches:
                continue
            ratio = random.choice(args.ratios)
            optimizer.zero_grad(set_to_none=True)
            metrics = {key: 0.0 for key in ("loss_3h", "loss_drop_6h", "loss_ce", "loss_full", "loss_feature")}
            for offset in range(0, len(images), args.micro_batch_size):
                x = images[offset:offset + args.micro_batch_size].cuda(non_blocking=True)
                y = targets[offset:offset + args.micro_batch_size].cuda(non_blocking=True)
                fraction = len(x) / len(images)
                layers = args.feature_layers if args.feature_weight else []
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        teacher.p = NARROW_P
                        teacher_features = teacher.forward_features(x, NARROW_P)
                        teacher_narrow = teacher.forward_head(teacher_features)
                        with capture_wide_features(teacher, layers) as teacher_tokens:
                            teacher_wide = wide_logits(teacher, wide_input(teacher, x, teacher_features))
                    features, narrow, attention = forward_narrow_with_attention(student, x, STAGES)
                    x0 = wide_input(student, x, features)
                    mask = make_update_mask(attention.float(), ratio, "topk")
                    with capture_wide_features(student, layers) as student_tokens:
                        wide = wide_logits(student, x0, mask)
                    full = wide_logits(student, x0) if args.full_weight else None
                loss_narrow, loss_wide = distill(narrow, teacher_narrow), distill(wide, teacher_wide)
                zero = narrow.new_zeros((), dtype=torch.float32)
                ce = 0.5 * (F.cross_entropy(narrow.float(), y) + F.cross_entropy(wide.float(), y)) if args.ce_weight else zero
                full_loss = distill(full, teacher_wide) if args.full_weight else zero
                feature_loss = aligned_feature_loss(student_tokens, teacher_tokens, mask) if args.feature_weight else zero
                loss = (0.5 * loss_narrow + 0.5 * loss_wide + args.ce_weight * ce
                        + args.full_weight * full_loss + args.feature_weight * feature_loss)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite loss at step {step}")
                (loss * fraction).backward()
                for key, value in zip(metrics, (loss_narrow, loss_wide, ce, full_loss, feature_loss)):
                    metrics[key] += value.detach().item() * fraction
                del loss, features, x0, wide, full, teacher_tokens, student_tokens
            grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            step += 1
            if step % 100 == 0 or step == total:
                log.append({"step": step, **metrics, "grad_norm": grad_norm.item(),
                            "ratio": ratio, "lr": schedule.get_last_lr()[0],
                            "images_per_s": (step - args.start_step) * args.batch_size / (time.time() - start)})
                print(json.dumps(log[-1]), flush=True)
                (output / "train_log.json").write_text(json.dumps(log) + "\n")
            if step % args.save_every == 0 or step == total:
                save()
            if step == total:
                break
    (output / "complete.json").write_text(json.dumps({"steps": step, "elapsed_seconds": time.time() - start,
                                                     "resume_augmentation_exact": False}, indent=2) + "\n")
    print(f"Done in {(time.time() - start) / 60:.1f} min: {output / 'student.pth'}", flush=True)


if __name__ == "__main__":
    main()
