from pathlib import Path
import sys
import unittest

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from drop_ft_losses import aligned_feature_loss, capture_wide_features


class DropFeatureLossTests(unittest.TestCase):
    def test_alignment_uses_original_positions_per_image_and_stops_teacher_gradients(self):
        teacher = torch.randn(2, 5, 8, requires_grad=True)
        mask = torch.tensor([[True, False, True, False, True], [True, True, False, True, False]])
        student = teacher.detach()[mask].reshape(2, 3, 8).clone().requires_grad_()
        loss = aligned_feature_loss({3: student}, {3: teacher}, mask)
        self.assertAlmostEqual(loss.item(), 0., places=7)
        changed = (student.detach() + torch.randn_like(student) * .2).requires_grad_()
        aligned_feature_loss({3: changed}, {3: teacher}, mask).backward()
        self.assertGreater(changed.grad.abs().sum().item(), 0.)
        self.assertIsNone(teacher.grad)

    def test_capture_preserves_gradients_and_always_removes_hooks(self):
        model = nn.Module()
        model.blocks = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        x = torch.randn(2, 3, 4, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            with capture_wide_features(model, [0, 1]) as features:
                y = model.blocks[1](model.blocks[0](x))
                self.assertIs(features[1], y)
                features[0].sum().backward()
                raise RuntimeError("interrupted")
        self.assertIsNotNone(x.grad)
        self.assertTrue(all(not block._forward_hooks for block in model.blocks))

    def test_unequal_token_counts_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal token count"):
            aligned_feature_loss({0: torch.randn(2, 2, 4)}, {0: torch.randn(2, 3, 4)},
                                 torch.tensor([[True, True, False], [True, False, False]]))


if __name__ == "__main__":
    unittest.main()
