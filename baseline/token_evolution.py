"""Evo-ViT Eq. 4--7 adapted to ThinkingViT's recycled 6H input.

Keep original spatial slots so placeholder tokens can be selected again. The
representative consumes one of the patch budget slots (equal attention lengths
to the drop control). This implements the paper's residual broadcast; official
DeiT code instead leaves placeholders unchanged, as its comment explains.
"""
from contextlib import contextmanager
import torch
from selective_refinement import cls_attention_from_qkv


METHODS = ("last_cls", "global_cls", "evo_layer", "evo_stage", "evo_schedule")


def selection_interval(method, progress=1.0):
    return 4 if method == "evo_stage" or (method == "evo_schedule" and progress >= 2 / 3) else 1


def gather(x, indices):
    return x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


@contextmanager
def block_attention(block):
    captured = []
    def hook(module, inputs, projected):
        with torch.no_grad():
            heads = projected.shape[-1] // (3 * block.attn.head_dim)
            captured.append(cls_attention_from_qkv(block.attn, projected.detach().float(), heads))
    handle = block.attn.qkv.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def split_indices(scores, count):
    selected = scores[:, 1:].topk(count, dim=1, sorted=False).indices + 1
    selected = selected.sort(dim=1).values
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[:, 0] = False
    mask.scatter_(1, selected, False)
    rest = torch.arange(scores.shape[1], device=scores.device).expand_as(scores)[mask].view(len(scores), -1)
    return selected, rest


def slow_fast_block(block, x, scores, selected, rest, decay=0.5):
    informative = gather(x, selected)
    placeholders = gather(x, rest)
    weights = scores.gather(1, rest)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    representative = (placeholders.float() * weights.unsqueeze(-1)).sum(1, keepdim=True).to(x.dtype)
    compact = torch.cat((x[:, :1], informative, representative), dim=1)
    with block_attention(block) as attention:
        evolved = block(compact, 1.0)
    # Sum of MSA and FFN residuals equals final output minus block input.
    delta = evolved[:, -1:] - representative
    updated = x.scatter(1, rest.unsqueeze(-1).expand_as(placeholders), placeholders + delta)
    updated = updated.scatter(1, selected.unsqueeze(-1).expand_as(informative), evolved[:, 1:-1])
    updated = torch.cat((evolved[:, :1], updated[:, 1:]), dim=1)
    # Only informative positions receive new attention; placeholders retain EMA.
    new = decay * scores.gather(1, selected) + (1 - decay) * attention[0][:, 1:-1]
    scores = scores.scatter(1, selected, new)
    return updated, scores


def evolution_logits(model, x, scores, ratio, method, progress=1.0, decay=0.5):
    if method not in METHODS or not 0 < ratio <= 1 or not 0 <= decay <= 1:
        raise ValueError("Invalid token evolution method, ratio or decay")
    model.p = 1.0
    budget = round((x.shape[1] - 1) * ratio)
    if budget < 1:
        raise ValueError("Token budget must leave at least one patch slot")
    if method in ("last_cls", "global_cls"):
        selected, _ = split_indices(scores, budget)
        x = torch.cat((x[:, :1], gather(x, selected)), dim=1)
        for block in model.blocks:
            x = block(x, 1.0)
    elif ratio == 1:
        for block in model.blocks:
            x = block(x, 1.0)
    else:
        interval = selection_interval(method, progress)
        for index, block in enumerate(model.blocks):
            if index % interval == 0:
                selected, rest = split_indices(scores, budget - 1)
            x, scores = slow_fast_block(block, x, scores, selected, rest, decay)
    return model.forward_head(model.norm(x, model.embed_dim))
