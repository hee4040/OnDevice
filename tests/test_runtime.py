from pathlib import Path
import sys
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from efficient_vit.model import linear
from efficient_vit.policy import PRESETS, Policy, entropy10, macs
from efficient_vit.runtime import Engine
from efficient_vit.verify import OriginalEngine


class EchoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))
        self.calls = []

    def narrow(self, images, return_embedding=False):
        b = len(images)
        return images, torch.zeros(b, 2), images, images[:, 0], images

    def wide(self, images, features, scores, ratio):
        self.calls.append((ratio, len(images)))
        return torch.full((len(images), 2), ratio)

    refine = wide


class RuntimeTests(unittest.TestCase):
    def test_main_baseline_exits_without_selector_and_retains_all_wide_tokens(self):
        class Native:
            def __init__(self):
                self.calls = []
            def forward_features(self, images, width, previous=None, stages=None):
                self.calls.append((width, len(images)))
                return images if width == .5 else images * 2
            def forward_head(self, features):
                return features
            def entropy(self, logits):
                return logits[:, :1]
        native = Native()  # No attention modules: this baseline must not access a selector.
        engine = OriginalEngine(native)
        output, codes, _ = engine.predict(torch.tensor([[.2, .3], [1., .4]]), "full")
        self.assertEqual(codes.tolist(), [-1, 0])
        self.assertEqual(native.calls, [(.5, 2), (1., 1)])
        torch.testing.assert_close(output, torch.tensor([[.2, .3], [2., .8]]))
        native.calls.clear()
        engine.predict(torch.tensor([[.2, .3]]), "full")
        self.assertEqual(native.calls, [(.5, 1)])
        self.assertEqual(macs(1, capture_cls=True)[0]-macs(1, capture_cls=False)[0], 197*192)

    def test_policy_boundaries_and_real_conditional_dispatch(self):
        model = EchoModel()
        engine = Engine(model)
        x = torch.tensor([[.79], [.8], [1.24], [1.25], [2.]])
        logits, codes, _ = engine.predict(x)
        self.assertEqual(codes.tolist(), [-1, 0, 0, 1, 1])
        self.assertEqual(model.calls, [(.25, 2), (.5, 2)])
        torch.testing.assert_close(logits[:, 0], torch.tensor([0., .25, .25, .5, .5]))
        model.calls.clear()
        engine.predict(torch.tensor([[.2]]))
        self.assertFalse(model.calls)

    def test_static_qkv_respects_three_projection_offsets(self):
        source = nn.Linear(4, 12)
        fixed = linear(source, 2, 6, qkv=True)
        rows = torch.tensor([0, 1, 4, 5, 8, 9])
        x = torch.randn(3, 2)
        expected = torch.nn.functional.linear(x, source.weight[rows, :2], source.bias[rows])
        torch.testing.assert_close(fixed(x), expected)

    def test_entropy_matches_official_categorical_and_budget_order(self):
        logits = torch.randn(4, 1000) * 10
        expected = torch.distributions.Categorical(probs=logits.softmax(-1).topk(10, dim=-1).values).entropy()
        torch.testing.assert_close(entropy10(logits), expected)
        self.assertTrue(macs(.25)[1] < macs(.5)[1] < macs(1)[1])
        self.assertEqual(macs(.5, cached_embedding=False)[1]-macs(.5)[1], 196*3*16*16*384)
        with self.assertRaises(ValueError):
            Policy(.8, (.7,), (.25, .5))
        with self.assertRaises(ValueError):
            Policy(.8, (), (.5, .25))


if __name__ == "__main__":
    unittest.main()
