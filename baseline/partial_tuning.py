"""Stage-specific PEFT: keep 3H unchanged and adapt only the Drop 6H pass."""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math

import torch
from torch import nn
import torch.nn.functional as F

MODES = ("p", "p_ln", "p_ln_lora")


@dataclass
class DropContext:
    active: bool = False


class DropLayerNorm(nn.Module):
    def __init__(self, original, context):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.eps = original.eps
        self.normalized_shape = original.normalized_shape
        self.context = context
        self.drop_weight = nn.Parameter(original.weight.detach().clone())
        self.drop_bias = nn.Parameter(original.bias.detach().clone())

    def forward(self, x, keep_shape):
        width = keep_shape or self.normalized_shape[0]
        use_drop = self.context.active and width == self.normalized_shape[0]
        weight = self.drop_weight if use_drop else self.weight
        bias = self.drop_bias if use_drop else self.bias
        return F.layer_norm(x[:, :, :width], (width,), weight[:width], bias[:width], self.eps)


class DropQKV(nn.Module):
    def __init__(self, original, context, rank=8, alpha=8.):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.context = context
        self.rank = rank
        self.scale = alpha / rank
        for part in ("q", "v"):
            a = nn.Parameter(self.weight.new_empty(rank, self.in_features))
            b = nn.Parameter(self.weight.new_zeros(self.in_features, rank))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            self.register_parameter(f"lora_{part}_a", a)
            self.register_parameter(f"lora_{part}_b", b)

    def forward(self, x, keep_shape):
        if keep_shape:
            indices = (list(range(keep_shape))
                       + list(range(self.out_features // 3, self.out_features // 3 + keep_shape))
                       + list(range(2 * self.out_features // 3, 2 * self.out_features // 3 + keep_shape)))
            result = F.linear(x[:, :, :keep_shape], self.weight[indices, :keep_shape], self.bias[indices])
        else:
            result = F.linear(x, self.weight, self.bias)
        if self.context.active and (keep_shape or x.shape[-1]) == self.in_features:
            q = F.linear(F.linear(x, self.lora_q_a), self.lora_q_b) * self.scale
            v = F.linear(F.linear(x, self.lora_v_a), self.lora_v_b) * self.scale
            result = result + torch.cat((q, torch.zeros_like(q), v), dim=-1)
        return result


def configure_partial(model, mode, rank=8):
    if mode not in MODES:
        raise ValueError(f"Unknown partial mode: {mode}")
    if hasattr(model, "_partial_context"):
        raise ValueError("Partial modules are already configured")
    if rank != 8:
        raise ValueError("This study fixes LoRA rank and alpha to 8")
    model.requires_grad_(False)
    model.proj_layer.requires_grad_(True)
    model.alpha[0].requires_grad_(True)
    context = DropContext()
    model._partial_context = context
    model._partial_mode = mode
    if mode != "p":
        for block in model.blocks:
            block.norm1 = DropLayerNorm(block.norm1, context)
            block.norm2 = DropLayerNorm(block.norm2, context)
        model.norm = DropLayerNorm(model.norm, context)
    if mode == "p_ln_lora":
        for block in model.blocks:
            block.attn.qkv = DropQKV(block.attn.qkv, context, rank=rank)
    return {"mode": mode, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "trainable_names": [name for name, p in model.named_parameters() if p.requires_grad],
            "rank": rank if mode == "p_ln_lora" else None}


def partial_mode_from_state(state):
    if any("lora_q_a" in key for key in state):
        return "p_ln_lora"
    if any("drop_weight" in key for key in state):
        return "p_ln"
    return None  # Projection-only uses the original inference architecture/state keys.


@contextmanager
def drop_context(model, active):
    context = getattr(model, "_partial_context", None)
    if context is None:
        yield
        return
    previous = context.active
    context.active = active
    try:
        yield
    finally:
        context.active = previous


def frozen_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def verify_narrow_preserved(student, teacher, images):
    from selective_refinement import forward_narrow_with_attention, make_update_mask
    with drop_context(student, False):
        _, actual, scores = forward_narrow_with_attention(student, images)
        _, expected, expected_scores = forward_narrow_with_attention(teacher, images)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(scores, expected_scores, rtol=0, atol=0)
    entropy = student.entropy(actual)
    expected_entropy = teacher.entropy(expected)
    torch.testing.assert_close(entropy, expected_entropy, rtol=0, atol=0)
    for ratio in (.25, .5, .75):
        if not torch.equal(make_update_mask(scores, ratio), make_update_mask(expected_scores, ratio)):
            raise AssertionError("Partial tuning changed 3H token selection")
    return {"logits_max_abs_diff": (actual - expected).abs().max().item(),
            "attention_max_abs_diff": (scores - expected_scores).abs().max().item(),
            "entropy_max_abs_diff": (entropy - expected_entropy).abs().max().item(),
            "routing_mismatches": int(((entropy >= 1) != (expected_entropy >= 1)).sum()),
            "token_selection_equal": True, "samples": len(images)}
