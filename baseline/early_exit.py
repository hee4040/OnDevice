"""Batch-1 early exit using the original ThinkingViT stages and entropy gate."""
import torch


def forward_early_exit(model, x, threshold=1.0):
    if model.training:
        raise ValueError("Early exit requires model.eval()")
    if x.shape[0] != 1:
        raise ValueError("This baseline supports batch size 1 only")
    stages = model._resolve_thinking_stages([3, 6])
    p = model.p = model._stage_ratio(stages[0])
    features = model.forward_features(x, p)
    logits = model.forward_head(features)
    entropy = model.entropy(logits)
    # This GPU-to-CPU synchronization is part of measured inference time.
    if (entropy < threshold).item():
        return logits, torch.zeros_like(entropy, dtype=torch.long)
    p = model.p = model._stage_ratio(stages[1])
    features = model.forward_features(x, p, features, stages)
    return model.forward_head(features), torch.ones_like(entropy, dtype=torch.long)


def forward_early_exit_batch(model, x, threshold=1.0):
    """Accuracy path: run the wide stage only on the uncertain batch subset."""
    if model.training:
        raise ValueError("Early exit requires model.eval()")
    stages = model._resolve_thinking_stages([3, 6])
    p = model.p = model._stage_ratio(stages[0])
    features = model.forward_features(x, p)
    logits = model.forward_head(features)
    stage = (~(model.entropy(logits) < threshold)).long()
    selected = stage[:, 0].nonzero(as_tuple=True)[0]
    if selected.numel() == 0:
        return logits, stage
    p = model.p = model._stage_ratio(stages[1])
    if selected.numel() == x.shape[0]:
        features = model.forward_features(x, p, features, stages)
        return model.forward_head(features), stage
    features = model.forward_features(x.index_select(0, selected), p,
                                      features.index_select(0, selected), stages)
    logits = logits.clone()
    logits.index_copy_(0, selected, model.forward_head(features))
    return logits, stage
