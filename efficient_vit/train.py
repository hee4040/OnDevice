"""The retained full-weight KD recipe; routing policies are applied after training."""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import time
import torch
from torch.nn import functional as F
from .model import ROOT, load_native


def student_narrow(model, images):
    attention = model.blocks[-1].attn
    scores = []
    def hook(module, args, projected):
        with torch.no_grad():
            b, n, _ = projected.shape
            q, k, _ = projected.detach().reshape(b, n, 3, 3, 64).permute(2, 0, 3, 1, 4).unbind(0)
            q, k = attention.q_norm(q[:, :, :1]), attention.k_norm(k)
            scores.append(((q * attention.scale) @ k.transpose(-2, -1)).softmax(-1).mean(1).squeeze(1))
    handle = attention.qkv.register_forward_hook(hook)
    try:
        model.p = .5
        features = model.forward_features(images, .5)
        return features, model.forward_head(features), scores
    finally:
        handle.remove()


def student_wide(model, images, features, scores, ratio):
    model.p = 1.
    x = model.norm_pre(model.patch_drop(model._pos_embed(model.patch_embed(images))))
    x = x + model.alpha[0] * model.proj_layer(features)
    keep = scores[:, 1:].topk(round(196 * ratio), dim=1).indices.sort(dim=1).values + 1
    keep = torch.cat((torch.zeros_like(keep[:, :1]), keep), dim=1)
    x = x.gather(1, keep[..., None].expand(-1, -1, 384))
    for block in model.blocks:
        x = block(x, 1.)
    return model.forward_head(model.norm(x, 384))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/imagenet1k")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/thinkingvit_3h6h_ema.pth")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.micro_batch) < 1 or args.workers < 0 or (args.max_steps is not None and args.max_steps < 1):
        parser.error("Invalid training sizes")
    if args.output.exists() and not args.resume:
        parser.error("Output exists; choose a new directory")
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True
    student = load_native(args.checkpoint, device="cuda")
    teacher = copy.deepcopy(student).requires_grad_(False)
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from timm.data import create_transform, resolve_data_config
    dataset = ImageFolder(args.data / "train", transform=create_transform(**resolve_data_config({}, model=student), is_training=True))
    if len(dataset) != 1281167:
        raise RuntimeError("Expected all 1,281,167 ImageNet training images")
    generator = torch.Generator().manual_seed(42)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=args.workers,
                        drop_last=True, pin_memory=True, persistent_workers=bool(args.workers))
    total = min(args.epochs * len(loader), args.max_steps or args.epochs * len(loader))
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s + 1) / 500) * .5 * (1 + math.cos(math.pi * min(s, total) / total)))
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cuda", weights_only=True)
        previous = state["arguments"]
        for key in ("epochs", "batch_size", "micro_batch", "max_steps", "data", "checkpoint"):
            if previous[key] != (str(getattr(args, key)) if isinstance(getattr(args, key), Path) else getattr(args, key)):
                raise ValueError(f"Cannot resume with changed {key}")
        student.load_state_dict(state["student"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    args.output.mkdir(parents=True, exist_ok=True)
    arguments = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    (args.output / "arguments.json").write_text(json.dumps(arguments, indent=2) + "\n")
    def save():
        torch.save(student.state_dict(), args.output / "student.tmp.pth")
        (args.output / "student.tmp.pth").replace(args.output / "student.pth")
        torch.save({"student": student.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "step": step, "arguments": arguments, "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()}, args.output / "resume.tmp.pth")
        (args.output / "resume.tmp.pth").replace(args.output / "resume.pth")
    def kd(output, target):
        return F.kl_div(output.float().log_softmax(-1), target.float().log_softmax(-1), reduction="batchmean", log_target=True)
    started = time.time()
    while step < total:
        epoch, skip = divmod(step, len(loader))
        generator.manual_seed(42 + epoch)
        for index, (images, _) in enumerate(loader):
            if index < skip:
                continue
            optimizer.zero_grad(set_to_none=True)
            ratio = random.choice((.25, .5))
            loss_value = 0.
            for offset in range(0, len(images), args.micro_batch):
                x = images[offset:offset + args.micro_batch].cuda(non_blocking=True)
                fraction = len(x) / len(images)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    with torch.no_grad():
                        teacher.p = .5
                        tf = teacher.forward_features(x, .5)
                        tn = teacher.forward_head(tf)
                        teacher.p = 1.
                        tw = teacher.forward_head(teacher.forward_features(x, 1., tf, (3, 6)))
                    features, narrow, scores = student_narrow(student, x)
                    wide = student_wide(student, x, features, scores[0], ratio)
                loss = .5 * kd(narrow, tn) + .5 * kd(wide, tw)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite KD loss")
                (loss * fraction).backward()
                loss_value += loss.item() * fraction
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 100 == 0 or step == total:
                print(json.dumps({"step": step, "total": total, "loss": loss_value, "ratio": ratio}), flush=True)
            if step % 500 == 0 or step == total:
                save()
            if step == total:
                break
    (args.output / "complete.json").write_text(json.dumps({"steps": step, "elapsed_seconds": time.time() - started,
        "worker_augmentation_resume_bit_exact": False}, indent=2) + "\n")


if __name__ == "__main__":
    main()
