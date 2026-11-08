"""Independent small-case checks for budget optimization and grouped inference."""
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
import adaptive_drop as runtime
import analyze_adaptive_drop as analysis


class AdaptiveDropTests(unittest.TestCase):
    def data(self, n, seed):
        rng = np.random.default_rng(seed)
        return {'ids': rng.permutation(n), 'entropy': rng.choice([.5, 1., 1.5, 2., 3.], n),
                'narrow': rng.integers(0, 2, n).astype(bool),
                'correct': rng.integers(0, 2, (n, 3)).astype(bool)}

    def test_oracle_equals_exhaustive_optimum_including_nonmonotonic_correctness(self):
        for seed in range(40):
            data = self.data(6, seed)
            routed = np.flatnonzero(data['entropy'] >= 1)
            budget = analysis.outcomes(data, analysis.fixed_codes(data))[1]
            best = 0
            for choices in itertools.product(range(3), repeat=len(routed)):
                codes = np.full(6, -1); codes[routed] = choices
                correct, macs = analysis.outcomes(data, codes)
                if macs <= budget:
                    best = max(best, int(correct.sum()))
            observed, spent = analysis.outcomes(data, analysis.oracle(data))
            self.assertEqual(int(observed.sum()), best)
            self.assertLessEqual(spent, budget)

    def test_policy_matches_direct_enumeration(self):
        for seed in range(5):
            data = self.data(15, seed)
            e = torch.from_numpy(data['entropy'][data['entropy'] >= 1])
            bounds = np.unique(np.r_[-np.inf, torch.quantile(e, torch.linspace(0, 1, 101, dtype=e.dtype)), np.inf])
            budget = analysis.outcomes(data, analysis.fixed_codes(data))[1]
            best = None
            for i, a in enumerate(bounds):
                for b in bounds[i:]:
                    codes = np.where(data['entropy'] < 1, -1,
                                     np.where(data['entropy'] < a, 0, np.where(data['entropy'] < b, 1, 2)))
                    correct, cost = analysis.outcomes(data, codes)
                    key = (-int(correct.sum()), cost, a, b)
                    if cost <= budget and (best is None or key < best):
                        best = key
            policy = analysis.select_policy(data)
            self.assertEqual((-policy['calibration_correct'], policy['calibration_total_macs'],
                              runtime.decode_boundary(policy['t1']), runtime.decode_boundary(policy['t2'])), best)

    def test_oracle_equal_cost_tie_uses_image_id(self):
        data = {'ids': np.array([20, 10, 30]), 'entropy': np.ones(3), 'narrow': np.zeros(3, dtype=bool),
                'correct': np.array([[False, False, True]] * 3)}
        self.assertEqual(analysis.oracle(data).tolist(), [0, 2, 0])

    def test_equal_thresholds_and_fixed_policy(self):
        e = torch.tensor([.9, 1., 1.5, 2., 3.])
        self.assertEqual(runtime.ratio_codes(e, {'t1': 1.5, 't2': 2.}).tolist(), [-1, 0, 1, 2, 2])
        self.assertEqual(runtime.ratio_codes(e, {'t1': '-inf', 't2': '+inf'}).tolist(), [-1, 1, 1, 1, 1])
        self.assertEqual(runtime.ratio_codes(e, {'t1': 2., 't2': 2.}).tolist(), [-1, 0, 0, 2, 2])
        for value in [float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                runtime.ratio_codes(torch.tensor([value]), {'t1': 1., 't2': 2.})

    def test_float64_quantile_boundary_is_not_rounded_to_float32(self):
        e = torch.tensor([1.5], dtype=torch.float32)
        boundary = float(e.item()) + 1e-9
        self.assertEqual(runtime.ratio_codes(e, {'t1': boundary, 't2': 2.}).item(), 0)

    def test_all_exits_and_nonmonotonic_group_order(self):
        for entropies in [[.1, .2, .3], [3., .5, 1.1, 2., 1.2], [3., 3.], [1.1]]:
            n = len(entropies)
            model = Mock(training=False)
            model._stage_ratio.side_effect = lambda heads: heads / 6
            model.entropy.return_value = torch.tensor(entropies)[:, None]
            x = torch.arange(n, dtype=torch.float32)[:, None]
            features, logits, attention = x + 100, x.clone(), torch.ones(n, 197)
            calls = []

            def fake_wide_input(m, images, f):
                torch.testing.assert_close(f, images + 100)
                return images

            def fake_logits(m, images, mask):
                calls.extend(images[:, 0].int().tolist())
                return images + mask + 10

            with patch.object(runtime, 'forward_narrow_with_attention', return_value=(features, logits, attention)), \
                 patch.object(runtime, 'wide_input', side_effect=fake_wide_input), \
                 patch.object(runtime, 'make_update_mask', side_effect=lambda att, r, mode: r), \
                 patch.object(runtime, 'wide_logits', side_effect=fake_logits):
                for policy in [{'t1': 1.5, 't2': 2.}] + [{'kind': 'fixed', 'ratio': r} for r in runtime.RATIOS]:
                    calls.clear()
                    output, codes, _ = runtime.forward_adaptive(model, x, policy)
                    expected = x.clone()
                    for i, code in enumerate(codes.tolist()):
                        if code >= 0:
                            expected[i] += 10 + runtime.RATIOS[code]
                    torch.testing.assert_close(output, expected)
                    self.assertEqual(sorted(calls), (codes >= 0).nonzero()[:, 0].tolist())

    def test_identified_cache_and_invalid_identity(self):
        cache = {key: torch.zeros(3, dtype=torch.bool) for key in analysis.KEYS}
        cache.update(entropy=torch.ones(3), image_id=torch.tensor([2, 0, 1]),
                     image_path=['c/c.jpg', 'a/a.jpg', 'b/b.jpg'], target=torch.tensor([2, 0, 1]))
        analysis.validate_cache(cache, 3)
        for ids in [torch.tensor([1, 1, 2]), torch.tensor([1, 2, 3])]:
            with self.assertRaises(ValueError):
                analysis.validate_cache(dict(cache, image_id=ids), 3)
        with self.assertRaises(KeyError):
            analysis.validate_cache({k: v for k, v in cache.items() if k != 'image_id'}, 3)

    def test_no_routed_images_and_paired_bootstrap(self):
        data = self.data(5, 0); data['entropy'][:] = .5
        policy = analysis.select_policy(data)
        self.assertEqual(analysis.apply_policy(data, policy).tolist(), [-1] * 5)
        self.assertEqual(analysis.paired_interval(data['narrow'], data['narrow'], repeats=20), [0., 0.])


if __name__ == '__main__':
    unittest.main()
