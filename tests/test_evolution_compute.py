from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline"))
from evolution_compute import average_macs, macs


class ComputeTests(unittest.TestCase):
    def test_routing_endpoints_and_unchanged_token_budget(self):
        narrow = macs("full")["narrow_core"]
        self.assertEqual(average_macs("full", 1, 0)[1], narrow / 1e9)
        full = macs("full")
        self.assertEqual(average_macs("full", 1, 1)[1], (narrow + full["wide_core"]) / 1e9)
        for ratio in (.25, .5, .75):
            base = macs("last_cls", ratio)
            for method in ("global_cls", "evo_layer", "evo_stage"):
                value = macs(method, ratio)
                self.assertEqual(value["wide_core"], base["wide_core"])
                self.assertEqual(value["wide_block_tokens"], base["wide_block_tokens"])
            self.assertGreater(average_macs("evo_stage", ratio, .36)[1], average_macs("last_cls", ratio, .36)[1])

    def test_actual_embedding_width_and_full_ratio_bypass(self):
        # Previous narrow-width embedding approximation undercounted 28.9M MACs.
        previous_narrow = 1253683200
        self.assertEqual(macs("full")["narrow_core"] - previous_narrow, 196 * 3 * 16 * 16 * 192)
        self.assertEqual(macs("evo_stage", 1)["wide_selection_aggregation"], 0)
        with self.assertRaises(ValueError):
            average_macs("full", 1, 1.1)


if __name__ == "__main__":
    unittest.main()
