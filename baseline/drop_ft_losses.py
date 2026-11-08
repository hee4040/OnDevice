"""Training-only losses for the drop fine-tuning ablations."""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


@contextmanager
def capture_wide_features(model, layers):
    """Capture block outputs with gradients; enter only around the 6H forward."""
    captured = {}
    handles = []
    for index in layers:
        def hook(module, inputs, output, index=index):
            captured[index] = output
        handles.append(model.blocks[index].register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


def aligned_feature_loss(student, teacher, mask):
    """Compare the same original patch positions, including CLS, at each layer.

    wide_blocks gathers tokens in original position order, not Top-K score order.
    Layer normalization removes arbitrary activation scales before MSE.
    """
    if not student or student.keys() != teacher.keys():
        raise ValueError("Student and teacher must contain the same nonempty feature layers")
    counts = mask.sum(1)
    if not torch.equal(counts, counts[:1].expand_as(counts)):
        raise ValueError("Feature alignment requires an equal token count per image")
    losses = []
    for index, actual in student.items():
        target = teacher[index].detach()
        target = target[mask].reshape(actual.shape)
        losses.append(F.mse_loss(F.layer_norm(actual.float(), (actual.shape[-1],)),
                                 F.layer_norm(target.float(), (target.shape[-1],))))
    return torch.stack(losses).mean()
