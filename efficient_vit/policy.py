from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class Policy:
    gate: float
    boundaries: tuple
    ratios: tuple

    def __post_init__(self):
        if not math.isfinite(self.gate) or self.gate < 0:
            raise ValueError("gate must be finite and nonnegative")
        if len(self.ratios) != len(self.boundaries) + 1 or not self.ratios:
            raise ValueError("One ratio is required per entropy interval")
        if any(not math.isfinite(x) for x in self.boundaries) or tuple(sorted(set(self.boundaries))) != self.boundaries:
            raise ValueError("boundaries must increase strictly")
        if self.boundaries and self.boundaries[0] <= self.gate:
            raise ValueError("First boundary must exceed exit gate")
        if any(r not in (.25, .5, .75, 1.) for r in self.ratios) or tuple(sorted(self.ratios)) != self.ratios:
            raise ValueError("Supported retention ratios must be monotone")

    def codes(self, entropy):
        result = torch.zeros_like(entropy, dtype=torch.long)
        for boundary in self.boundaries:
            result += (entropy.double() >= boundary).long()
        return torch.where(entropy.double() < self.gate, -1, result)


# Locked before evaluating this integration: transferred from target80-results.
# No claim that a policy calibrated on another checkpoint retains exactly 80%.
PRESETS = {
    "economy": Policy(1., (), (.25,)),
    "balanced": Policy(.8, (1.25,), (.25, .5)),
    "accuracy": Policy(.75, (1.15, 1.75), (.25, .5, .75)),
    "kd50": Policy(1., (), (.5,)),
    "full": Policy(1., (), (1.,)),
}


def entropy10(logits):
    probabilities = logits.softmax(-1).topk(10, dim=-1).values
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    eps = torch.finfo(probabilities.dtype).eps
    return -(probabilities * probabilities.clamp(eps, 1 - eps).log()).sum(-1)


def macs(ratio, cached_embedding=True):
    """Actual executed dimensions; matmul/conv MACs, not memory or kernel costs."""
    patches, tokens, d, wide = 196, 197, 192, 384
    patch = patches * 3 * 16 * 16 * wide
    block = lambda n, c: 12 * n * c * c + 2 * n * n * c
    narrow = patch + 12 * block(tokens, d) + d * 1000
    n = 1 + round(patches * ratio)
    stage2 = (0 if cached_embedding else patch) + tokens * d * wide + 12 * block(n, wide) + wide * 1000
    # Last narrow CLS query is extracted for selection, even for the full control.
    return narrow + tokens * d, stage2
