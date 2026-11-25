"""One selected 6H drop path per routed image; shared model instances are not thread-safe."""
import math
import torch
from evaluate_drop import STAGES, wide_logits
from evaluate_h1_diagnosis import wide_input
from selective_refinement import forward_narrow_with_attention, make_update_mask

RATIOS = (.25, .5, .75)


def decode_boundary(value):
    return {'-inf': -math.inf, '+inf': math.inf}.get(value, value) if isinstance(value, str) else float(value)


def ratio_codes(entropy, policy):
    """-1=3H exit, 0=25%, 1=50%, 2=75%; equal thresholds belong to the higher interval."""
    if not torch.isfinite(entropy).all():
        raise ValueError('Nonfinite entropy')
    # Calibration quantiles are float64; rounding a boundary to float32 can move
    # an image to the adjacent interval even when its entropy itself is unchanged.
    entropy = entropy.double()
    routed = entropy >= 1.0
    if policy.get('kind') == 'fixed':
        ratio = float(policy['ratio'])
        if ratio not in RATIOS:
            raise ValueError('Unsupported ratio')
        codes = torch.full_like(entropy, RATIOS.index(ratio), dtype=torch.long)
    else:
        t1, t2 = decode_boundary(policy['t1']), decode_boundary(policy['t2'])
        if not isinstance(t1, (int, float)) or not isinstance(t2, (int, float)) or math.isnan(t1) or math.isnan(t2) or t1 > t2:
            raise ValueError('Invalid policy boundaries')
        codes = torch.where(entropy < t1, 0, torch.where(entropy < t2, 1, 2)).long()
    return torch.where(routed, codes, -1)


def forward_adaptive(model, images, policy):
    """Batch-1 or grouped-batch inference, restoring original image order."""
    if model.training:
        raise ValueError('model.eval() required')
    model.p = model._stage_ratio(3)
    features, logits, attention = forward_narrow_with_attention(model, images, STAGES)
    entropy = model.entropy(logits)[:, 0]
    codes = ratio_codes(entropy, policy)
    output = logits.clone()
    for code, ratio in enumerate(RATIOS):
        selected = (codes == code).nonzero(as_tuple=True)[0]
        if selected.numel() == 0:
            continue
        model.p = model._stage_ratio(6)
        x0 = wide_input(model, images.index_select(0, selected), features.index_select(0, selected))
        mask = make_update_mask(attention.index_select(0, selected), ratio, 'topk')
        output.index_copy_(0, selected, wide_logits(model, x0, mask))
    return output, codes, entropy
