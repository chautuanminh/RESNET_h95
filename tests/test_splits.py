import csv
import tempfile
import unittest
from pathlib import Path

from src.splits import FINAL_EVAL_DATASETS, SplitPolicyError, create_or_load_split


class SplitTests(unittest.TestCase):
    def test_deterministic_split_reuses_saved_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            indices = list(range(120))

            first = create_or_load_split(indices, out, seed=42, val_count=10)
            second = create_or_load_split(list(reversed(indices)), out, seed=42, val_count=10)

            self.assertEqual(first.train_indices, second.train_indices)
            self.assertEqual(first.val_indices, second.val_indices)
            self.assertEqual(len(first.val_indices), 10)
            self.assertEqual(len(first.train_indices), 110)

            with (out / "splits" / "train_indices_seed42.csv").open(newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual([int(row["dataset_index"]) for row in rows], first.train_indices)

    def test_train_val_have_no_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            split = create_or_load_split(range(50), Path(tmp), seed=42, val_count=7)
            self.assertEqual(set(split.train_indices) & set(split.val_indices), set())

    def test_official_eval_folders_are_blocked_from_training(self):
        self.assertIn("DocTamperV1-TestingSet", FINAL_EVAL_DATASETS)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SplitPolicyError):
                create_or_load_split(
                    range(20),
                    Path(tmp),
                    seed=42,
                    val_count=5,
                    source_folder="DocTamperV1-TestingSet",
                )


if __name__ == "__main__":
    unittest.main()
