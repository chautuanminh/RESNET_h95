import unittest

from src.tampering_type_analysis import _summary


class TamperSummaryTests(unittest.TestCase):
    def test_summary_metrics_match_grouped_counts(self):
        rows = [
            {"test_set": "TestingSet", "tamper_type": "copy_move", "tp": 2, "fp": 1, "fn": 1, "tn": 6, "f1": 0.6667, "iou": 0.5, "precision": 0.6667, "recall": 0.6667, "mask_area_ratio": 0.3, "predicted_area_ratio": 0.3, "error_area_ratio": 0.2},
            {"test_set": "TestingSet", "tamper_type": "copy_move", "tp": 1, "fp": 0, "fn": 1, "tn": 8, "f1": 0.6667, "iou": 0.5, "precision": 1.0, "recall": 0.5, "mask_area_ratio": 0.2, "predicted_area_ratio": 0.1, "error_area_ratio": 0.1},
        ]
        summary = _summary(rows, ["test_set", "tamper_type"])[0]
        self.assertEqual(summary["tp"], 3)
        self.assertEqual(summary["fp"], 1)
        self.assertEqual(summary["fn"], 2)
        self.assertEqual(summary["tn"], 14)
        self.assertAlmostEqual(summary["precision"], 0.75)
        self.assertAlmostEqual(summary["recall"], 0.6)


if __name__ == "__main__":
    unittest.main()
