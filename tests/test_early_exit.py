from pathlib import Path
import sys
import unittest
from unittest.mock import Mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from early_exit import forward_early_exit, forward_early_exit_batch


class EarlyExitTests(unittest.TestCase):
    def model(self):
        model = Mock(training=False)
        model._resolve_thinking_stages.return_value = (3, 6)
        model._stage_ratio.side_effect = lambda heads: heads / 6
        model.forward_features.side_effect = [torch.ones(1, 2, 3), torch.ones(1, 2, 6)]
        model.forward_head.side_effect = [torch.tensor([[2., 1.]]), torch.tensor([[1., 3.]])]
        model.entropy.return_value = torch.tensor([[1.]])
        return model

    def test_easy_image_never_calls_wide_stage(self):
        model = self.model()
        logits, stage = forward_early_exit(model, torch.zeros(1, 3, 4, 4), threshold=1.1)
        self.assertEqual(model.forward_features.call_count, 1)
        self.assertEqual(model.forward_head.call_count, 1)
        self.assertEqual(stage.item(), 0)
        self.assertEqual(logits.argmax(1).item(), 0)

    def test_threshold_equality_enters_wide_stage_and_recycles_features(self):
        model = self.model()
        x = torch.zeros(1, 3, 4, 4)
        logits, stage = forward_early_exit(model, x, threshold=1.0)
        self.assertEqual(model.forward_features.call_count, 2)
        args = model.forward_features.call_args.args
        self.assertIs(args[0], x)
        self.assertEqual(args[1], 1.0)
        self.assertIs(args[2], model.forward_head.call_args_list[0].args[0])
        self.assertEqual(args[3], (3, 6))
        self.assertEqual(model.p, 1.0)
        self.assertEqual(stage.item(), 1)
        self.assertEqual(logits.argmax(1).item(), 1)

    def test_training_mode_rejected(self):
        model = self.model()
        model.training = True
        with self.assertRaises(ValueError):
            forward_early_exit(model, torch.zeros(1, 3, 4, 4))
        model.forward_features.assert_not_called()

    def test_multi_image_batch_rejected(self):
        model = self.model()
        with self.assertRaises(ValueError):
            forward_early_exit(model, torch.zeros(2, 3, 4, 4))
        model.forward_features.assert_not_called()

    def test_mixed_batch_refines_only_uncertain_images_in_original_order(self):
        model = self.model()
        x = torch.arange(4.).view(4, 1)
        narrow_features = x + 10
        narrow_logits = torch.tensor([[4., 1.], [3., 1.], [2., 1.], [1., 0.]])
        model.forward_features.side_effect = [narrow_features, torch.ones(2, 6)]
        model.forward_head.side_effect = [narrow_logits, torch.tensor([[0., 5.], [0., 6.]])]
        model.entropy.return_value = torch.tensor([[.1], [1.2], [.2], [1.]])
        logits, stage = forward_early_exit_batch(model, x, threshold=1.0)
        wide_args = model.forward_features.call_args.args
        torch.testing.assert_close(wide_args[0], x[[1, 3]])
        torch.testing.assert_close(wide_args[2], narrow_features[[1, 3]])
        torch.testing.assert_close(logits, torch.tensor([[4., 1.], [0., 5.], [2., 1.], [0., 6.]]))
        self.assertEqual(stage[:, 0].tolist(), [0, 1, 0, 1])

    def test_uniform_batch_skips_or_reuses_full_inputs(self):
        for threshold, expected_calls in ((10.0, 1), (0.0, 2)):
            with self.subTest(threshold=threshold):
                model = self.model()
                x = torch.zeros(3, 2)
                model.forward_features.side_effect = [x, x]
                model.forward_head.side_effect = [x, x]
                model.entropy.return_value = torch.ones(3, 1)
                _, stage = forward_early_exit_batch(model, x, threshold)
                self.assertEqual(model.forward_features.call_count, expected_calls)
                self.assertIs(model.forward_features.call_args.args[0], x)
                self.assertEqual(stage.sum().item(), 3 * (expected_calls - 1))


if __name__ == "__main__":
    unittest.main()
