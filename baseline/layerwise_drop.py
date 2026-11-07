"""Nested token dropping in 6H, ranked once by final 3H CLS attention.

Counts are absolute fractions of the original patches, not of surviving tokens.
Drop happens BEFORE each block. Removed tokens never return; CLS always survives.
"""
import math

import torch

PATCHES, DEPTH, NARROW, WIDE = 196, 12, 192, 384


def token_counts(ratios, patches=PATCHES, depth=DEPTH):
    if len(ratios) != depth or any(not math.isfinite(r) or not 0 <= r <= 1 for r in ratios):
        raise ValueError(f"Expected {depth} finite ratios in [0, 1]")
    if any(b > a for a, b in zip(ratios, ratios[1:])):
        raise ValueError("Drop schedules must be non-increasing; removed tokens cannot return")
    return tuple(1 + round(patches * r) for r in ratios)


def block_macs(tokens, width):
    return 12 * tokens * width ** 2 + 2 * tokens ** 2 * width


def stage_macs(counts, width):
    # Same convention as plot_h1_diagnosis.py: patch embedding, blocks, CLS head.
    return PATCHES * 3 * 16 * 16 * width + sum(block_macs(n, width) for n in counts) + width * 1000


NARROW_MACS = stage_macs([197] * DEPTH, NARROW)


def wide_macs(counts):
    return 197 * NARROW * WIDE + stage_macs(counts, WIDE)


def experiment_configs():
    """Predeclared schedules and nearest-cost uniform controls, without validation tuning."""
    schedules = {"full": [1.] * DEPTH}
    for final in (0.25, 0.5, 0.75):
        tag = round(final * 100)
        schedules[f"uniform_{tag}"] = [final] * DEPTH
        schedules[f"staged_{tag}"] = [1.] * 4 + [(1 + final) / 2] * 4 + [final] * 4
        schedules[f"linear_{tag}"] = [1 - (1 - final) * i / 11 for i in range(DEPTH)]
        schedules[f"late_{tag}"] = [1.] * 8 + [final] * 4  # existing full-k=8 control
    rows = []
    for name, ratios in list(schedules.items()):
        counts = token_counts(ratios)
        if name.startswith(("staged_", "linear_", "late_")):
            k = min(range(197), key=lambda k: abs(wide_macs([k + 1] * DEPTH) - wide_macs(counts)))
            control = f"uniform_n{k}"
            schedules.setdefault(control, [k / PATCHES] * DEPTH)
        else:
            control = None
        rows.append(dict(name=name, ratios=ratios, counts=list(counts), matched_uniform=control))
    for name, ratios in schedules.items():
        if name not in {r["name"] for r in rows}:
            rows.append(dict(name=name, ratios=ratios, counts=list(token_counts(ratios)), matched_uniform=None))
    return rows


def drop_blocks(model, x, scores, ratios):
    """Physically gather nested Top-K subsets, preserving original spatial order."""
    if model.training:
        raise ValueError("Layer-wise drop requires model.eval()")
    counts = token_counts(ratios, x.shape[1] - 1, len(model.blocks))
    if scores.shape != x.shape[:2]:
        raise ValueError("Attention scores must match the original sequence")
    # Compute ranks once. topk uses the same selection as the preceding drop experiment.
    order = scores[:, 1:].topk(x.shape[1] - 1, dim=1, sorted=True).indices
    ranks = torch.empty_like(order).scatter_(1, order, torch.arange(order.shape[1], device=x.device).expand_as(order))
    ranks = torch.cat([torch.full_like(ranks[:, :1], -1), ranks], 1)
    model.p = 1.0
    for block, count in zip(model.blocks, counts):
        if count < x.shape[1]:
            positions = (ranks < count - 1).nonzero(as_tuple=False)[:, 1].view(len(x), count)
            x = x.gather(1, positions.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
            ranks = ranks.gather(1, positions)
        x = block(x, 1.0)
    return x


def layerwise_logits(model, x0, scores, ratios):
    x = drop_blocks(model, x0, scores, ratios)
    return model.forward_head(model.norm(x, model.embed_dim))
