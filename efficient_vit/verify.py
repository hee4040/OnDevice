"""Native model and static/graph implementation equivalence checks."""
import torch


@torch.inference_mode()
def native_narrow(native, images):
    captured = []
    attention = native.blocks[-1].attn
    def hook(module, args, projected):
        b, n, _ = projected.shape
        q, k, _ = projected.reshape(b, n, 3, 3, 64).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = attention.q_norm(q[:, :, :1]), attention.k_norm(k)
        captured.append(((q * attention.scale) @ k.transpose(-2, -1)).softmax(-1).mean(1).squeeze(1))
    handle = attention.qkv.register_forward_hook(hook)
    try:
        native.p = .5
        features = native.forward_features(images, .5)
        logits = native.forward_head(features)
    finally:
        handle.remove()
    return features, logits, captured[0], native.entropy(logits).flatten()


@torch.inference_mode()
def native_wide(native, images, features, scores, ratio):
    native.p = 1.
    x = native.norm_pre(native.patch_drop(native._pos_embed(native.patch_embed(images))))
    x = x + native.alpha[0] * native.proj_layer(features)
    if ratio < 1:
        selected = scores[:, 1:].topk(round(196 * ratio), dim=1).indices + 1
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[:, 0] = True
        mask.scatter_(1, selected, True)
        keep = mask.nonzero()[:, 1].view(len(x), -1)
        x = x.gather(1, keep[..., None].expand(-1, -1, 384))
    for block in native.blocks:
        x = block(x, 1.)
    return native.forward_head(native.norm(x, 384))


@torch.inference_mode()
def verify(native, model, images, graph=None):
    reference = native_narrow(native, images)
    actual = model.narrow(images)
    tolerance = dict(rtol=1e-4, atol=2e-5) if images.dtype == torch.float32 else dict(rtol=.03, atol=.02)
    errors = {}
    for name, left, right in zip(("features", "logits", "attention", "entropy"), actual, reference):
        torch.testing.assert_close(left, right, **tolerance)
        errors[name] = (left - right).abs().max().item()
    for ratio in (.25, .5, .75, 1.):
        expected = native_wide(native, images, reference[0], reference[2], ratio)
        output = model.wide(images, actual[0], actual[2], ratio)
        torch.testing.assert_close(output, expected, **tolerance)
        errors[f"wide_{ratio}"] = (output - expected).abs().max().item()
    if graph:
        from .runtime import Engine
        eager = Engine(model)
        for policy in ("economy", "balanced", "accuracy", "kd50", "full"):
            expected = eager.predict(images, policy)
            output = graph.predict(images, policy)
            torch.testing.assert_close(output[0], expected[0], **tolerance)
            torch.testing.assert_close(output[1], expected[1], rtol=0, atol=0)
            errors[f"graph_{policy}"] = (output[0] - expected[0]).abs().max().item()
    return errors


class NativeEngine:
    """Unoptimized original-code control for same-session measurements only."""
    def __init__(self, native):
        self.native = native

    @torch.inference_mode()
    def predict(self, images, policy, narrow_result=None):
        from .policy import PRESETS
        policy = PRESETS[policy] if isinstance(policy, str) else policy
        features, logits, scores, entropy = native_narrow(self.native, images) if narrow_result is None else narrow_result
        if len(images) != 1:
            codes = policy.codes(entropy)
            output = logits.clone()
            for code, ratio in enumerate(policy.ratios):
                selected = (codes == code).nonzero(as_tuple=True)[0]
                if len(selected):
                    wide = native_wide(self.native, images[selected], features[selected], scores[selected], ratio)
                    output.index_copy_(0, selected, wide)
            return output, codes, entropy
        e = entropy.item()
        code = -1 if e < policy.gate else sum(e >= b for b in policy.boundaries)
        if code >= 0:
            logits = native_wide(self.native, images, features, scores, policy.ratios[code])
        return logits, torch.tensor([code], device=images.device), entropy


class OriginalEngine:
    """Original paper's full-token 3H -> entropy gate -> 6H, without CLS hooks."""
    def __init__(self, native):
        self.native = native

    @torch.inference_mode()
    def narrow(self, images):
        self.native.p = .5
        features = self.native.forward_features(images, .5)
        logits = self.native.forward_head(features)
        return features, logits, None, self.native.entropy(logits).flatten()

    @torch.inference_mode()
    def predict(self, images, policy, narrow_result=None):
        from .policy import PRESETS
        policy = PRESETS[policy] if isinstance(policy, str) else policy
        if policy.ratios != (1.,) or policy.gate != 1.:
            raise ValueError("Original ThinkingViT control requires full tokens and tau=1")
        features, logits, _, entropy = self.narrow(images) if narrow_result is None else narrow_result
        if len(images) == 1:
            if (entropy < 1.).item():
                return logits, torch.full_like(entropy, -1, dtype=torch.long), entropy
            self.native.p = 1.
            wide = self.native.forward_features(images, 1., features, (3, 6))
            return self.native.forward_head(wide), torch.zeros_like(entropy, dtype=torch.long), entropy
        selected = (~(entropy < 1.)).nonzero(as_tuple=True)[0]
        codes = torch.where(entropy < 1., -1, 0).long()
        if not len(selected):
            return logits, codes, entropy
        self.native.p = 1.
        if len(selected) == len(images):
            wide = self.native.forward_features(images, 1., features, (3, 6))
            return self.native.forward_head(wide), codes, entropy
        wide = self.native.forward_features(images[selected], 1., features[selected], (3, 6))
        result = logits.clone()
        result.index_copy_(0, selected, self.native.forward_head(wide))
        return result, codes, entropy
