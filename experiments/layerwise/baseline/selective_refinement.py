"""Accuracy oracle for selective token refinement in ThinkingViT.

This module intentionally computes dense 6H blocks and masks their updates.  It
tests whether selective updates preserve accuracy; it does not claim a latency
or FLOPs reduction.  A later sparse implementation can use the same semantics.
"""
from contextlib import contextmanager

import torch


def cls_attention_from_qkv(attention, projected, heads):
    """Return head-mean CLS attention without changing fused attention execution."""
    batch, tokens, _ = projected.shape
    qkv = projected.reshape(batch, tokens, 3, heads, attention.head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    q, k = attention.q_norm(q[:, :, :1]), attention.k_norm(k)
    probabilities = ((q * attention.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
    return probabilities.mean(dim=1).squeeze(1)


@contextmanager
def capture_narrow_cls_attention(model, narrow_heads):
    """Capture final-block attention only for the narrow forward."""
    captured = []
    attention = model.blocks[-1].attn

    def hook(module, inputs, projected):
        heads = projected.shape[-1] // (3 * attention.head_dim)
        if heads == narrow_heads:
            captured.append(cls_attention_from_qkv(attention, projected.detach(), heads))

    handle = attention.qkv.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def forward_narrow_with_attention(model, x, stages=(3, 6)):
    """Run 3H once and return features, logits, and final-block CLS attention."""
    if model.training:
        raise ValueError("Selective refinement requires model.eval()")
    stages = model._resolve_thinking_stages(stages)
    if len(stages) != 2:
        raise ValueError("Selective refinement currently supports exactly two stages")
    p = model.p = model._stage_ratio(stages[0])
    with capture_narrow_cls_attention(model, stages[0]) as captured:
        features = model.forward_features(x, p)
    if len(captured) != 1:
        raise RuntimeError(f"Expected one narrow attention tensor, captured {len(captured)}")
    return features, model.forward_head(features), captured[0]


def make_update_mask(scores, ratio, policy="topk", generator=None):
    """Build a fixed token mask; CLS is always selected and ratio applies to patches."""
    if scores.ndim != 2 or scores.shape[1] < 2:
        raise ValueError("scores must have shape [batch, CLS + patch tokens]")
    if not 0.0 <= ratio <= 1.0:
        raise ValueError("ratio must be between 0 and 1")
    if policy not in {"topk", "bottomk", "random"}:
        raise ValueError(f"Unknown selection policy: {policy}")
    batch, tokens = scores.shape
    patches = tokens - 1
    count = round(patches * ratio)
    mask = torch.zeros((batch, tokens), dtype=torch.bool, device=scores.device)
    mask[:, 0] = True
    if count == 0:
        return mask
    if policy == "random":
        ranking_scores = torch.rand((batch, patches), generator=generator, device="cpu").to(scores.device)
        largest = True
    else:
        ranking_scores = scores[:, 1:]
        largest = policy == "topk"
    selected = ranking_scores.topk(count, dim=1, largest=largest, sorted=False).indices + 1
    mask.scatter_(1, selected, True)
    return mask


def forward_selective_wide(model, images, narrow_features, update_mask, stages=(3, 6)):
    """Run the dense-mask 6H oracle with full K/V context and frozen non-selected tokens."""
    if model.training:
        raise ValueError("Selective refinement requires model.eval()")
    stages = model._resolve_thinking_stages(stages)
    if len(stages) != 2:
        raise ValueError("Selective refinement currently supports exactly two stages")
    p = model.p = model._stage_ratio(stages[1])
    x = model.patch_embed(images)
    x = model._pos_embed(x)
    x = model.patch_drop(x)
    x = model.norm_pre(x)
    x = x + model.alpha[0] * model.proj_layer(narrow_features)
    if update_mask.shape != x.shape[:2]:
        raise ValueError(f"update_mask shape {tuple(update_mask.shape)} does not match tokens {tuple(x.shape[:2])}")
    mask = update_mask.unsqueeze(-1)
    for block in model.blocks:
        candidate = block(x, p)
        x = torch.where(mask, candidate, x)
    x = model.norm(x, int(p * model.embed_dim))
    return x


def selective_logits(model, images, narrow_features, attention_scores, ratio, policy="topk",
                     generator=None, stages=(3, 6)):
    """Select tokens and return 6H logits plus the applied update mask."""
    mask = make_update_mask(attention_scores, ratio, policy, generator)
    features = forward_selective_wide(model, images, narrow_features, mask, stages)
    return model.forward_head(features), mask
