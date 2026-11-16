from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_imagenet import check_counts, split_files


class PreparationTests(unittest.TestCase):
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
