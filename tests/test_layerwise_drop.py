from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from layerwise_drop import drop_blocks, experiment_configs, token_counts, wide_macs
from evaluate_h1_diagnosis import wide_blocks
from selective_refinement import make_update_mask
from evaluate_layerwise_drop import summarize, pipeline
from unittest.mock import Mock, patch


class RecordingBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seen = None

    def forward(self, x, p):
        self.seen = x.clone()
        return x + x.mean(1, keepdim=True)


class DropTests(unittest.TestCase):
    def model(self, depth=12):
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList([RecordingBlock() for _ in range(depth)])
        return model.eval()

    def test_absolute_counts_token_identity_and_cls(self):
        model = self.model(3)
        x = torch.arange(10.).reshape(2, 5, 1)
        scores = torch.tensor([[0., 1., 4., 3., 2.], [0., 4., 1., 2., 3.]])
        result = drop_blocks(model, x, scores, [1, .5, 0])
        self.assertEqual([b.seen.shape[1] for b in model.blocks], [5, 3, 1])
        first = x + x.mean(1, keepdim=True)
        expected = torch.stack([first[0, [0, 2, 3]], first[1, [0, 1, 4]]])
        torch.testing.assert_close(model.blocks[1].seen, expected)
        self.assertEqual(result.shape, (2, 1, 1))

    def test_uniform_matches_existing_drop(self):
        torch.manual_seed(2)
        x, scores = torch.randn(3, 197, 4), torch.rand(3, 197)
        for ratio in (0, .25, .5, .75, 1):
            model = self.model()
            actual = drop_blocks(model, x, scores, [ratio] * 12)
            expected = wide_blocks(model, x, 1., make_update_mask(scores, ratio), 'drop')
            torch.testing.assert_close(actual, expected)

    def test_invalid_schedules_and_training(self):
        for ratios in ([.5, 1], [float('nan')] * 12, [1.1] * 12, [1] * 11, [float('inf')] * 12, [.5, .75] + [.5] * 10):
            with self.assertRaises(ValueError):
                token_counts(ratios)
        self.assertEqual(token_counts([0] * 12), (1,) * 12)
        with self.assertRaises(ValueError):
            drop_blocks(self.model().train(), torch.ones(1, 197, 1), torch.ones(1, 197), [1.] * 12)

    def test_cost_convention_and_nearest_controls(self):
        from plot_h1_diagnosis import wide_macs as previous_macs
        rows = experiment_configs()
        for ratio in (.25, .5, .75, 1):
            self.assertEqual(wide_macs(token_counts([ratio] * 12)), previous_macs(dict(ratio=ratio, mode='drop', k=0)))
        for row in rows:
            if row['matched_uniform']:
                control = next(c for c in rows if c['name'] == row['matched_uniform'])
                delta = abs(wide_macs(row['counts']) - wide_macs(control['counts']))
                self.assertEqual(delta, min(abs(wide_macs(row['counts']) - wide_macs([n] * 12)) for n in range(1, 198)))

    def test_easy_answers_preserved_and_no_routing(self):
        configs = [c for c in experiment_configs() if c['name'] in ('full', 'uniform_25')]
        data = dict(entropy=torch.tensor([0., 2., 3.]), narrow_correct=torch.tensor([True, False, True]),
                    full_correct=torch.tensor([False, True, True]), uniform_25_correct=torch.tensor([False, True, False]))
        summary = summarize(data, configs, 1.)
        self.assertAlmostEqual(summary['rows'][1]['top1'], 200 / 3)
        summary = summarize(data, configs, 10.)
        self.assertIsNone(summary['rows'][0]['top1_routed'])
        self.assertEqual(summary['rows'][0]['routed'], 0)

    def test_pipeline_skips_easy_and_routes_equality(self):
        model = Mock()
        triple = (torch.ones(1, 2, 2), torch.ones(1, 2), torch.ones(1, 2))
        config = dict(name='uniform_50', ratios=[.5] * 12)
        with patch('evaluate_layerwise_drop.forward_narrow_with_attention', return_value=triple), \
             patch('evaluate_layerwise_drop.wide_input') as prepare, \
             patch('evaluate_layerwise_drop.layerwise_logits') as wide:
            model.entropy.return_value = torch.tensor(.9)
            self.assertFalse(pipeline(model, None, config, 1.)[1])
            wide.assert_not_called()
            prepare.assert_not_called()
            model.entropy.return_value = torch.tensor(1.)
            self.assertTrue(pipeline(model, None, config, 1.)[1])
            wide.assert_called_once()


if __name__ == '__main__':
    unittest.main()
