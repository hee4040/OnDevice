from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from analyze_stages import cls_attention_from_qkv, summarize, transition_group


class StageAnalysisTests(unittest.TestCase):
    def test_all_transition_groups(self):
        self.assertEqual(transition_group(True, True), "both_correct")
        self.assertEqual(transition_group(False, True), "corrected")
        self.assertEqual(transition_group(True, False), "regressed")
        self.assertEqual(transition_group(False, False), "both_wrong")

    def test_summary_separates_counterfactual_wide_from_routed_accuracy(self):
        rows = [
            {"group": "corrected", "narrow_correct": 0, "wide_correct": 1, "routed_to_6h": 0, "routed_correct": 0},
            {"group": "regressed", "narrow_correct": 1, "wide_correct": 0, "routed_to_6h": 1, "routed_correct": 0},
            {"group": "both_correct", "narrow_correct": 1, "wide_correct": 1, "routed_to_6h": 0, "routed_correct": 1},
            {"group": "both_wrong", "narrow_correct": 0, "wide_correct": 0, "routed_to_6h": 1, "routed_correct": 0},
        ]
        result = summarize(rows)
        self.assertEqual(result["all"]["narrow_top1"], 50)
        self.assertEqual(result["all"]["wide_top1"], 50)
        self.assertEqual(result["all"]["routed_top1"], 25)
        self.assertEqual(result["routed_to_6h"]["net_corrected_images"], -1)
        self.assertEqual(result["exited_at_3h"]["groups"]["corrected"], 1)
        self.assertIsNone(summarize([])["all"]["narrow_top1"])

    def test_cls_row_matches_full_attention_including_self(self):
        attention = SimpleNamespace(head_dim=4, scale=0.5, q_norm=torch.nn.LayerNorm(4), k_norm=torch.nn.LayerNorm(4))
        projected = torch.randn(2, 5, 36, generator=torch.Generator().manual_seed(42))
        q, k, _ = projected.reshape(2, 5, 3, 3, 4).permute(2, 0, 3, 1, 4).unbind(0)
        full = ((attention.q_norm(q) * .5) @ attention.k_norm(k).transpose(-2, -1)).softmax(-1)
        actual = cls_attention_from_qkv(attention, projected, heads=3)
        torch.testing.assert_close(actual, full[:, :, 0, :].mean(1))
        torch.testing.assert_close(actual.sum(1), torch.ones(2))
        self.assertTrue((actual[:, 1:].sum(1) < 1).all())


if __name__ == "__main__":
    unittest.main()
