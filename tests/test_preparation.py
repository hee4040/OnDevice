import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline import parse_log
from prepare_imagenet import check_counts, split_files


class PreparationTests(unittest.TestCase):
    def test_complete_result_ignores_intermediate_accuracy(self):
        log = ('Test: Acc@1 1.0 Acc@5 2.0\n'
               'Model 1 Dispatched: 60.00% (30000/50000)\n'
               'Model 2 Dispatched: 40.00% (20000/50000)\n'
               '--result\n' + json.dumps({"top1": 80.31, "top5": 95.0}))
        result = parse_log(log)
        self.assertEqual(result["top1"], 80.31)
        self.assertEqual(result["stage_2_fraction"], 0.4)
        self.assertAlmostEqual(result["gmacs_official_routing_estimate"], 3.09)

    def test_partial_accuracy_is_not_reported_as_complete(self):
        with self.assertRaises(ValueError):
            parse_log('Test: Acc@1 80.0 Acc@5 95.0')
        with self.assertRaises(ValueError):
            parse_log('Model 1 Dispatched: 50.00% (250/500)\n'
                      'Model 2 Dispatched: 50.00% (250/500)\n--result {"top1":80}')

    def test_incomplete_download_listing_is_rejected(self):
        with self.assertRaises(ValueError):
            split_files(["data/train-00000-of-00002.parquet"], "train")
        with self.assertRaises(ValueError):
            split_files(["data/train-00001-of-00002.parquet", "data/train-00002-of-00002.parquet"], "train")
        files = ["data/train-00001-of-00002.parquet", "data/train-00000-of-00002.parquet", "data/test-00000-of-00001.parquet"]
        self.assertEqual(split_files(files, "train"), sorted(files[:2]))

    def test_missing_class_cannot_pass_by_total_count_alone(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "validation"
            (folder / "0000").mkdir(parents=True)
            (folder / "0000" / "image.JPEG").write_bytes(b"test fixture")
            with self.assertRaises(ValueError):
                check_counts(folder, expected=1)


if __name__ == "__main__":
    unittest.main()
