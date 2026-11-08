from pathlib import Path
import sys
import itertools
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from evaluate_nonmonotone_entropy import fit_binned, predict_binned, stratified_paired_ci
from analyze_adaptive_drop import outcomes, fixed_codes


class NonmonotoneTests(unittest.TestCase):
    def test_optimizer_matches_direct_exhaustive_predictions(self):
        for seed in range(10):
            rng = np.random.default_rng(seed)
            d = {'ids': np.arange(15), 'entropy': rng.choice([.5, 1., 1.1, 1.5, 2., 3.], 15),
                 'correct': rng.integers(0, 2, (15, 3)).astype(bool), 'narrow': rng.integers(0, 2, 15).astype(bool)}
            for monotone in (False, True):
                p = fit_binned(d, 3, monotone)
                budget = outcomes(d, fixed_codes(d))[1]; best = None
                for codes in itertools.product(range(3), repeat=3):
                    if monotone and list(codes) != sorted(codes): continue
                    corr, cost = outcomes(d, predict_binned(d['entropy'], dict(p, codes=codes)))
                    key = (-int(corr.sum()), cost, codes)
                    if cost <= budget and (best is None or key < best): best = key
                self.assertEqual((-p['train_correct'], p['train_total_macs'], tuple(p['codes'])), best)

    def test_boundary_exit_and_nonmonotonic_order(self):
        p = {'cuts': [1.5, 2.], 'codes': [0, 2, 1]}
        self.assertEqual(predict_binned(np.array([.9, 1., 1.5, 1.9, 2., 3.]), p).tolist(), [-1, 0, 2, 2, 1, 1])

    def test_empty_routed_set(self):
        d = {'entropy': np.array([.2, .3]), 'correct': np.zeros((2, 3), dtype=bool), 'narrow': np.ones(2, dtype=bool)}
        p = fit_binned(d)
        self.assertEqual(predict_binned(d['entropy'], p).tolist(), [-1, -1])

    def test_paired_interval_preserves_fold_sizes_and_pairing(self):
        a = np.array([True] * 5 + [False] * 5); b = ~a
        folds = np.array([0] * 5 + [1] * 5)
        self.assertEqual(stratified_paired_ci(a, b, folds, repeats=100), [0., 0.])
        self.assertEqual(stratified_paired_ci(a, a, folds, repeats=100), [0., 0.])


if __name__ == '__main__':
    unittest.main()
