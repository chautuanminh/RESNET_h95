import unittest

from src.metrics import (
    StreamingBinaryMetrics,
    ThresholdSweep,
    confusion_from_flat,
    metrics_from_counts,
)


class MetricsTests(unittest.TestCase):
    def test_streaming_counts_match_naive_counts(self):
        probs_batches = [[0.1, 0.8, 0.6], [0.2, 0.9, 0.4]]
        labels_batches = [[0, 1, 0], [0, 1, 1]]

        stream = StreamingBinaryMetrics(threshold=0.5)
        all_probs = []
        all_labels = []
        for probs, labels in zip(probs_batches, labels_batches):
            stream.update(probs, labels)
            all_probs.extend(probs)
            all_labels.extend(labels)

        expected = confusion_from_flat(all_probs, all_labels, threshold=0.5)
        self.assertEqual(stream.counts, expected)
        self.assertAlmostEqual(metrics_from_counts(stream.counts)["f1"], 2 / 3)

    def test_threshold_sweep_accumulates_counts(self):
        sweep = ThresholdSweep([0.25, 0.5, 0.75])
        sweep.update([0.2, 0.4, 0.8], [0, 1, 1])
        rows = {row["threshold"]: row for row in sweep.rows()}

        self.assertEqual(rows[0.25]["tp"], 2)
        self.assertEqual(rows[0.25]["fp"], 0)
        self.assertEqual(rows[0.5]["tp"], 1)
        self.assertEqual(rows[0.5]["fn"], 1)
        self.assertEqual(rows[0.75]["tp"], 1)
        self.assertEqual(rows[0.75]["fp"], 0)


if __name__ == "__main__":
    unittest.main()
