from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from mlp_selective_refinement import (
    forward_full_attention_selective_mlp_block,
    forward_mlp_selective_wide,
    make_mlp_update_mask,
)


class IdentityNorm(torch.nn.Module):
    def forward(self, x, keep_shape):
        return x[:, :, :keep_shape]


class IdentityLayerScale(torch.nn.Module):
    def forward(self, x, keep_shape):
        return x[:, :, :keep_shape]


class ConstantUpdate(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x, p):
        return torch.full_like(x, self.value)


class ToyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 4
        self.norm1 = IdentityNorm()
        self.norm2 = IdentityNorm()
        self.attn = ConstantUpdate(1.0)
        self.mlp = ConstantUpdate(2.0)
        self.ls1 = IdentityLayerScale()
        self.ls2 = IdentityLayerScale()
        self.drop_path1 = torch.nn.Identity()
        self.drop_path2 = torch.nn.Identity()

    def forward(self, x, p):
        keep_shape = int(p * self.dim)
        x = x[:, :, :keep_shape] + self.attn(self.norm1(x, keep_shape), p)
        return x[:, :, :keep_shape] + self.mlp(self.norm2(x, keep_shape), p)


class ToyWideModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Identity()
        self.patch_drop = torch.nn.Identity()
        self.norm_pre = torch.nn.Identity()
        self.blocks = torch.nn.ModuleList([ToyBlock(), ToyBlock()])
        self.alpha = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1))])
        self.proj_layer = torch.nn.Linear(2, 4, bias=False)
        self.embed_dim = 4
        self.p = None

    def _resolve_thinking_stages(self, stages):
        return tuple(stages)

    def _stage_ratio(self, heads):
        return heads / 6

    def _pos_embed(self, x):
        return x

    def norm(self, x, keep_shape):
        return x[:, :, :keep_shape]


class MlpSelectiveRefinementTests(unittest.TestCase):
    def test_masks_keep_cls_and_exact_patch_count(self):
        scores = torch.tensor([[9.0, 0.1, 0.8, 0.2, 0.7]])
        top = make_mlp_update_mask(scores, .5, "topk")
        bottom = make_mlp_update_mask(scores, .5, "bottomk")
        random = make_mlp_update_mask(scores, .5, "random", torch.Generator().manual_seed(7))
        self.assertEqual(top.tolist(), [[True, False, True, False, True]])
        self.assertEqual(bottom.tolist(), [[True, True, False, True, False]])
        self.assertEqual(int(random.sum()), 3)
        self.assertTrue(bool(random[0, 0]))

    def test_zero_ratio_updates_only_cls_mlp(self):
        mask = make_mlp_update_mask(torch.ones(2, 5), 0, "topk")
        self.assertEqual(mask.tolist(), [[True, False, False, False, False]] * 2)

    def test_same_random_seed_produces_nested_ratio_sweep(self):
        scores = torch.ones(3, 9)
        small = make_mlp_update_mask(scores, .25, "random", torch.Generator().manual_seed(17))
        large = make_mlp_update_mask(scores, .75, "random", torch.Generator().manual_seed(17))
        self.assertTrue(bool((small & ~large).sum() == 0))

    def test_attention_updates_all_tokens_and_mlp_only_selected_tokens(self):
        block = ToyBlock().eval()
        x = torch.zeros(1, 4, 4)
        mask = torch.tensor([[True, False, True, False]])
        actual = forward_full_attention_selective_mlp_block(block, x, 1.0, mask)
        expected = torch.tensor([[[3., 3., 3., 3.], [1., 1., 1., 1.],
                                  [3., 3., 3., 3.], [1., 1., 1., 1.]]])
        torch.testing.assert_close(actual, expected)

    def test_full_mask_matches_original_block(self):
        block = ToyBlock().eval()
        x = torch.randn(2, 4, 4)
        mask = torch.ones(2, 4, dtype=torch.bool)
        actual = forward_full_attention_selective_mlp_block(block, x, 1.0, mask)
        torch.testing.assert_close(actual, block(x, 1.0))

    def test_unselected_tokens_accumulate_attention_across_blocks(self):
        model = ToyWideModel().eval()
        images = torch.zeros(1, 4, 4)
        narrow = torch.zeros(1, 4, 2)
        mask = torch.tensor([[True, False, True, False]])
        actual = forward_mlp_selective_wide(model, images, narrow, mask)
        expected = torch.tensor([[[6., 6., 6., 6.], [2., 2., 2., 2.],
                                  [6., 6., 6., 6.], [2., 2., 2., 2.]]])
        torch.testing.assert_close(actual, expected)

    def test_invalid_ratio_and_policy_are_rejected(self):
        scores = torch.ones(1, 5)
        for ratio in (-.1, 1.1):
            with self.assertRaises(ValueError):
                make_mlp_update_mask(scores, ratio)
        with self.assertRaises(ValueError):
            make_mlp_update_mask(scores, .5, "unknown")


if __name__ == "__main__":
    unittest.main()
