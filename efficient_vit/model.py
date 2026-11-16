"""Convert custom dynamic-width layers into ordinary, fixed-width PyTorch layers."""
import copy
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_native(checkpoint, weights=None, repo=None, device="cpu"):
    repo = Path(repo or ROOT / "third_party/ThinkingViT").resolve()
    sys.path.insert(0, str(repo))
    import timm
    from validate import _load_eval_checkpoint, RLDropping
    if not Path(timm.__file__).resolve().is_relative_to(repo):
        raise RuntimeError("ThinkingViT's custom timm is required")
    model = timm.create_model("thinkingvit", pretrained=False, thinking_stages=(3, 6))
    _load_eval_checkpoint(model, str(checkpoint), use_ema=True)
    RLDropping(model, used_heads=6, max_heads=6, ema=False)
    if weights:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state.get("student", state))
    if model.embed_dim != 384 or model.patch_embed.num_patches != 196 or model.num_prefix_tokens != 1:
        raise ValueError("Expected ThinkingViT 3H/6H, 224px, one CLS")
    return model.to(device).eval()


def linear(source, inputs, outputs, qkv=False):
    result = nn.Linear(inputs, outputs, bias=source.bias is not None)
    if qkv:
        base = source.weight.shape[0] // 3
        width = outputs // 3
        indices = torch.cat([torch.arange(i * base, i * base + width) for i in range(3)]).to(source.weight.device)
        weight = source.weight.index_select(0, indices)[:, :inputs]
        bias = source.bias.index_select(0, indices) if source.bias is not None else None
    else:
        weight = source.weight[:outputs, :inputs]
        bias = source.bias[:outputs] if source.bias is not None else None
    with torch.no_grad():
        result.weight.copy_(weight)
        if bias is not None:
            result.bias.copy_(bias)
    return result


def norm(source, width):
    result = nn.LayerNorm(width, eps=source.eps)
    with torch.no_grad():
        result.weight.copy_(source.weight[:width])
        result.bias.copy_(source.bias[:width])
    return result


class Block(nn.Module):
    def __init__(self, source, width, capture=False):
        super().__init__()
        self.heads, self.width, self.capture = width // 64, width, capture
        self.norm1, self.norm2 = norm(source.norm1, width), norm(source.norm2, width)
        self.qkv = linear(source.attn.qkv, width, width * 3, True)
        self.q_norm, self.k_norm = copy.deepcopy(source.attn.q_norm), copy.deepcopy(source.attn.k_norm)
        self.proj = linear(source.attn.proj, width, width)
        self.fc1 = linear(source.mlp.fc1, width, 4 * width)
        self.fc2 = linear(source.mlp.fc2, 4 * width, width)
        self.act = copy.deepcopy(source.mlp.act)
        self.register_buffer("scale1", source.ls1.gamma[:width].detach().clone() if hasattr(source.ls1, "gamma") else torch.ones(width))
        self.register_buffer("scale2", source.ls2.gamma[:width].detach().clone() if hasattr(source.ls2, "gamma") else torch.ones(width))

    def forward(self, x):
        batch, tokens, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(batch, tokens, 3, self.heads, 64).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attention = ((q[:, :, :1] * .125) @ k.transpose(-2, -1)).softmax(-1).mean(1).squeeze(1) if self.capture else None
        values = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch, tokens, self.width)
        x = x + self.proj(values) * self.scale1
        x = x + self.fc2(self.act(self.fc1(self.norm2(x)))) * self.scale2
        return (x, attention) if self.capture else x


class Specialized(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.patch_embed = copy.deepcopy(native.patch_embed)
        self.register_buffer("cls_token", native.cls_token.detach().clone())
        self.register_buffer("pos_embed", native.pos_embed.detach().clone())
        if native.no_embed_class or not isinstance(native.norm_pre, nn.Identity):
            raise ValueError("Unsupported embedding configuration")
        self.narrow_blocks = nn.ModuleList([Block(b, 192, capture=i == 11) for i, b in enumerate(native.blocks)])
        self.wide_blocks = nn.ModuleList([Block(b, 384) for b in native.blocks])
        self.narrow_norm, self.wide_norm = norm(native.norm, 192), norm(native.norm, 384)
        self.narrow_head, self.wide_head = linear(native.head, 192, 1000), linear(native.head, 384, 1000)
        self.recycle = copy.deepcopy(native.proj_layer)
        self.register_buffer("alpha", native.alpha[0].detach().clone())

    def embed(self, images):
        patches = self.patch_embed(images)
        return torch.cat((self.cls_token.expand(len(images), -1, -1), patches), dim=1) + self.pos_embed

    def narrow(self, images, return_embedding=False):
        from .policy import entropy10
        embedding = self.embed(images)
        x = embedding[:, :, :192]
        for block in self.narrow_blocks[:-1]:
            x = block(x)
        x, scores = self.narrow_blocks[-1](x)
        features = self.narrow_norm(x)
        logits = self.narrow_head(features[:, 0])
        result = features, logits, scores, entropy10(logits)
        return (*result, embedding) if return_embedding else result

    def wide(self, images, features, scores, ratio):
        return self.refine(self.embed(images), features, scores, ratio)

    def refine(self, embedding, features, scores, ratio):
        x = embedding + self.alpha * self.recycle(features)
        if ratio < 1:
            selected = scores[:, 1:].topk(round(196 * ratio), dim=1).indices.sort(dim=1).values + 1
            keep = torch.cat((torch.zeros_like(selected[:, :1]), selected), dim=1)
            x = x.gather(1, keep.unsqueeze(-1).expand(-1, -1, 384))
        for block in self.wide_blocks:
            x = block(x)
        return self.wide_head(self.wide_norm(x)[:, 0])


def specialize(native, device="cuda", dtype=torch.float32):
    return Specialized(native).to(device=device, dtype=dtype).eval().requires_grad_(False)
