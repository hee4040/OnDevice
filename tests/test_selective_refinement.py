from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from selective_refinement import forward_selective_wide, make_update_mask


class AddOneBlock(torch.nn.Module):
    def forward(self, x, p):
        return x + 1


class IdentityWideModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Identity()
        self.patch_drop = torch.nn.Identity()
        self.norm_pre = torch.nn.Identity()
        self.blocks = torch.nn.ModuleList([AddOneBlock(), AddOneBlock()])
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
        return x


class SelectiveRefinementTests(unittest.TestCase):
    def test_top_bottom_and_random_masks_keep_cls_and_exact_count(self):
        scores = torch.tensor([[9.0, 0.1, 0.8, 0.2, 0.7]])
        top = make_update_mask(scores, .5, "topk")
        bottom = make_update_mask(scores, .5, "bottomk")
        random = make_update_mask(scores, .5, "random", torch.Generator().manual_seed(7))
        self.assertEqual(top.tolist(), [[True, False, True, False, True]])
        self.assertEqual(bottom.tolist(), [[True, True, False, True, False]])
        self.assertEqual(int(random.sum()), 3)
        self.assertTrue(bool(random[0, 0]))

    def test_zero_ratio_updates_only_cls(self):
        mask = make_update_mask(torch.ones(2, 5), 0, "topk")
        self.assertEqual(mask.tolist(), [[True, False, False, False, False]] * 2)

    def test_same_random_seed_produces_nested_ratio_sweep(self):
        scores = torch.ones(3, 9)
        small = make_update_mask(scores, .25, "random", torch.Generator().manual_seed(17))
        large = make_update_mask(scores, .75, "random", torch.Generator().manual_seed(17))
        self.assertTrue(bool((small & ~large).sum() == 0))

    def test_full_ratio_matches_dense_blocks(self):
        model = IdentityWideModel().eval()
        images = torch.zeros(1, 4, 4)
        narrow = torch.zeros(1, 4, 2)
        mask = torch.ones(1, 4, dtype=torch.bool)
        actual = forward_selective_wide(model, images, narrow, mask)
        torch.testing.assert_close(actual, torch.full_like(images, 2))

    def test_unselected_tokens_are_frozen_across_blocks(self):
        model = IdentityWideModel().eval()
        images = torch.zeros(1, 4, 4)
        narrow = torch.zeros(1, 4, 2)
        mask = torch.tensor([[True, False, True, False]])
        actual = forward_selective_wide(model, images, narrow, mask)
        expected = torch.tensor([[[2., 2., 2., 2.], [0., 0., 0., 0.],
                                  [2., 2., 2., 2.], [0., 0., 0., 0.]]])
        torch.testing.assert_close(actual, expected)

    def test_invalid_ratio_and_policy_are_rejected(self):
        scores = torch.ones(1, 5)
        for ratio in (-.1, 1.1):
            with self.assertRaises(ValueError):
                make_update_mask(scores, ratio)
        with self.assertRaises(ValueError):
            make_update_mask(scores, .5, "unknown")


if __name__ == "__main__":
    unittest.main()
