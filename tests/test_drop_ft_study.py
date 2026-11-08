from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from run_drop_ft_study import choose_components, half_scores, timing_report


class StudySelectionTests(unittest.TestCase):
    def test_timing_does_not_double_count_overlapping_work(self):
        state = {"completed_at": "2026-11-06T12:00:30+00:00",
                 "data_preparation": {"started": "2026-11-06T12:00:00+00:00",
                                      "finished": "2026-11-06T12:00:20+00:00"},
                 "commands": [{"log": "results/kd/train.log", "started": "2026-11-06T12:00:10+00:00",
                               "finished": "2026-11-06T12:00:25+00:00", "outcome": "실패"}]}
        report = "\n".join(timing_report(state))
        self.assertIn("**00:00:30**", report)
        self.assertIn("**00:00:05**", report)
        self.assertIn("11-06 21:00:00", report)
        self.assertIn("실패", report)

    def test_combination_never_uses_held_out_results(self):
        def score(drop, full, held_out):
            return {"selection": {"drop_50": drop, "full_6h": full},
                    "held_out": {"drop_50": held_out}}
        scores = {"kd": score(79., 81., 79.), "ce": score(79.3, 81., 0.),
                  "full": score(78.95, 81.3, 0.), "feature": score(79., 81., 100.)}
        self.assertEqual(choose_components(scores), ["ce", "full"])
        scores["ce"]["held_out"]["drop_50"] = 100.
        self.assertEqual(choose_components(scores), ["ce", "full"])

    def test_each_models_own_routing_and_fixed_halves(self):
        values = {"entropy": torch.cat((torch.zeros(25000), torch.ones(25000))),
                  "narrow_correct": torch.ones(50000, dtype=torch.bool),
                  "drop_50_correct": torch.zeros(50000, dtype=torch.bool),
                  "wide_correct": torch.ones(50000, dtype=torch.bool)}
        result = half_scores(values)
        self.assertEqual(result["selection"]["drop_50"], 100.)
        self.assertEqual(result["held_out"]["drop_50"], 0.)
        self.assertEqual(result["held_out"]["full_6h"], 100.)


if __name__ == "__main__":
    unittest.main()
