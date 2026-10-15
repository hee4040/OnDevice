import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'baseline'))
from audit_failures import original_id, assess, metrics


class FailureAuditTests(unittest.TestCase):
    def test_original_filename_id_not_dataset_index(self):
        self.assertEqual(original_id('ILSVRC2012_val_00017592_n02100735.JPEG'), (17592, 'n02100735'))
        self.assertEqual(original_id('ILSVRC2012_val_00000001_n02100735.JPEG')[0] - 1, 0)
        self.assertEqual(original_id('ILSVRC2012_val_00050000_n02100735.JPEG')[0] - 1, 49999)
        for name in ['validation-00000-of-00014-000001.JPEG', 'ILSVRC2012_val_00000000_n02100735.JPEG', 'ILSVRC2012_val_00050001_n02100735.JPEG']:
            with self.assertRaises(ValueError):
                original_id(name)

    def test_reassessment_can_accept_both_or_reverse_original_regression(self):
        row = {'narrow_prediction': '2', 'wide_prediction': '7', 'routed_prediction': '7'}
        self.assertEqual(assess(row, [2, 7])['real_group'], 'both_correct')
        self.assertEqual(assess(row, [7])['real_group'], 'corrected')
        self.assertEqual(assess(row, [2])['real_group'], 'regressed')
        self.assertEqual(assess(row, [3])['real_group'], 'both_wrong')
        self.assertIsNone(assess(row, [])['real_wide_correct'])

    def test_empty_annotations_excluded_and_original_denominators_kept(self):
        rows = []
        for labels in ([7], []):
            row = {'narrow_prediction': '2', 'wide_prediction': '7', 'routed_prediction': '2',
                   'narrow_correct': '1', 'wide_correct': '0', 'routed_correct': '1', 'group': 'regressed'}
            rows.append({**row, **assess(row, labels)})
        result = metrics(rows)
        self.assertEqual(result['samples'], 2)
        self.assertEqual(result['real_evaluated'], 1)
        self.assertEqual(result['real_excluded'], 1)
        self.assertEqual(result['wide']['real_top1'], 100)
        self.assertEqual(result['narrow']['original_correct_all'], 2)
        self.assertEqual(result['narrow']['original_correct_matched'], 1)
        self.assertEqual(result['routed']['real_top1'], 0)
        self.assertIsNone(metrics([])['wide']['real_top1'])


if __name__ == '__main__':
    unittest.main()
