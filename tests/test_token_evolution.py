from pathlib import Path
import sys
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from token_evolution import split_indices, slow_fast_block, selection_interval, block_attention


class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.head_dim, self.attn.scale = 2, 2 ** -0.5
        self.attn.q_norm, self.attn.k_norm = nn.Identity(), nn.Identity()
        self.attn.qkv = nn.Linear(2, 6, bias=False)
        nn.init.zeros_(self.attn.qkv.weight)

    def forward(self, x, p):
        self.attn.qkv(x)
        return x * 2 + 3


class EvolutionTests(unittest.TestCase):
    def test_original_positions_and_reentry(self):
        scores = torch.tensor([[0., .1, .8, .3], [0., .9, .2, .1]])
        selected, rest = split_indices(scores, 1)
        self.assertEqual(selected.tolist(), [[2], [1]])
        self.assertEqual(rest.tolist(), [[1, 3], [2, 3]])
        scores[0, 1] = 1
        self.assertEqual(split_indices(scores, 1)[0][0].item(), 1)

    def test_residual_broadcast_not_representative_replacement_and_gradients(self):
        x = torch.arange(8.).view(1, 4, 2).requires_grad_()
        scores = torch.tensor([[0., .1, .8, .3]])
        selected, rest = split_indices(scores, 1)
        output, attention = slow_fast_block(FakeBlock(), x, scores, selected, rest)
        rep = x[:, 1:2] * .25 + x[:, 3:4] * .75
        torch.testing.assert_close(output[:, 1], x[:, 1] + rep[:, 0] + 3)
        torch.testing.assert_close(output[:, 3], x[:, 3] + rep[:, 0] + 3)
        torch.testing.assert_close(output[:, 2], x[:, 2] * 2 + 3)
        torch.testing.assert_close(output[:, 0], x[:, 0] * 2 + 3)
        self.assertEqual(attention[0, 1].item(), scores[0, 1].item())
        self.assertAlmostEqual(attention[0, 2].item(), .4 + 1 / 6, places=6)
        output.sum().backward()
        self.assertTrue(torch.all(x.grad > 0))
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_stage_switch_and_hook_cleanup(self):
        self.assertEqual(selection_interval("evo_schedule", .66), 1)
        self.assertEqual(selection_interval("evo_schedule", 2 / 3), 4)
        self.assertEqual(selection_interval("evo_layer", 1), 1)
        block = FakeBlock()
        with self.assertRaises(RuntimeError):
            with block_attention(block):
                raise RuntimeError("test")
        self.assertFalse(block.attn.qkv._forward_hooks)


if __name__ == "__main__":
    unittest.main()
