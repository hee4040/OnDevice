from pathlib import Path
import sys
import unittest

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from partial_tuning import (DropContext, DropLayerNorm, DropQKV, configure_partial,
                            drop_context, frozen_digest, partial_mode_from_state)
from run_partial_ft_study import choose_learning_rate


def toy_model():
    model = nn.Module()
    model.proj_layer = nn.Linear(4, 8)
    model.alpha = nn.ParameterList([nn.Parameter(torch.ones(1))])
    block = nn.Module()
    block.norm1, block.norm2 = nn.LayerNorm(8), nn.LayerNorm(8)
    block.attn = nn.Module()
    block.attn.qkv = nn.Linear(8, 24)
    model.blocks = nn.ModuleList([block])
    model.norm = nn.LayerNorm(8)
    return model


class PartialTuningTests(unittest.TestCase):
    def test_learning_rate_uses_teacher_kl_not_calibration_accuracy(self):
        rows = [{"lr": 5e-5, "kd_routed": .2, "top1": 99.},
                {"lr": .001, "kd_routed": .1, "top1": 1.},
                {"lr": .0002, "kd_routed": .1, "top1": 0.},
                {"lr": .01, "kd_routed": float("nan"), "top1": 100.}]
        self.assertEqual(choose_learning_rate(rows), .0002)

    def test_only_intended_parameters_train_and_counts_match(self):
        for mode, count in (("p", 41), ("p_ln", 89), ("p_ln_lora", 345)):
            with self.subTest(mode=mode):
                model = toy_model()
                result = configure_partial(model, mode)
                self.assertEqual(result["trainable_parameters"], count)
                for name, p in model.named_parameters():
                    if p.requires_grad:
                        self.assertTrue(name.startswith(("proj_layer.", "alpha.")) or
                                        any(key in name for key in ("drop_weight", "drop_bias", "lora_")))

    def test_norm_uses_keep_shape_not_unsliced_input_width(self):
        original = nn.LayerNorm(8)
        context = DropContext(True)
        module = DropLayerNorm(original, context)
        with torch.no_grad():
            module.drop_bias.fill_(2.)
        x = torch.randn(2, 3, 8)
        # First narrow block still receives an 8D input, but normalizes only 4D.
        expected = F.layer_norm(x[:, :, :4], (4,), original.weight[:4], original.bias[:4], original.eps)
        torch.testing.assert_close(module(x, 4), expected, rtol=0, atol=0)
        torch.testing.assert_close(module(x, 8), original(x) + 2.)
        context.active = False
        torch.testing.assert_close(module(x, 8), original(x), rtol=0, atol=0)

    def test_lora_updates_only_qv_in_active_wide_path(self):
        original = nn.Linear(8, 24)
        context = DropContext(False)
        module = DropQKV(original, context)
        x = torch.randn(2, 3, 8)
        expected = original(x)
        torch.testing.assert_close(module(x, 8), expected, rtol=0, atol=0)
        with torch.no_grad():
            module.lora_q_b.fill_(.25)
            module.lora_v_b.fill_(.25)
        torch.testing.assert_close(module(x, 8), expected, rtol=0, atol=0)
        context.active = True
        actual = module(x, 8)
        self.assertFalse(torch.equal(actual[:, :, :8], expected[:, :, :8]))
        torch.testing.assert_close(actual[:, :, 8:16], expected[:, :, 8:16], rtol=0, atol=0)
        indices = list(range(4)) + list(range(8, 12)) + list(range(16, 20))
        narrow = F.linear(x[:, :, :4], original.weight[indices, :4], original.bias[indices])
        torch.testing.assert_close(module(x, 4), narrow, rtol=0, atol=0)

    def test_context_restores_after_exception_and_state_roundtrips(self):
        model = toy_model()
        configure_partial(model, "p_ln_lora")
        before = frozen_digest(model)
        with self.assertRaises(RuntimeError):
            with drop_context(model, True):
                self.assertTrue(model._partial_context.active)
                raise RuntimeError("interrupted")
        self.assertFalse(model._partial_context.active)
        clone = toy_model()
        configure_partial(clone, partial_mode_from_state(model.state_dict()))
        clone.load_state_dict(model.state_dict(), strict=True)
        self.assertEqual(frozen_digest(clone), before)
        with torch.no_grad():
            model.proj_layer.weight.add_(1.)
        self.assertEqual(frozen_digest(model), before)


if __name__ == "__main__":
    unittest.main()
