import tempfile
import unittest
from pathlib import Path

from src.visualization import save_training_curves


class TrainingCurveVisualizationTests(unittest.TestCase):
    def test_save_training_curves_writes_png_from_metric_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "plots" / "training_curves.png"
            train_rows = [
                {
                    "epoch": 1,
                    "train_loss": 0.8,
                    "val_loss": 0.7,
                    "train_f1": 0.2,
                    "val_f1": 0.3,
                    "train_images_per_sec": 10.0,
                    "val_images_per_sec": 8.0,
                    "lr": 0.001,
                },
                {
                    "epoch": 2,
                    "train_loss": 0.5,
                    "val_loss": 0.55,
                    "train_f1": 0.45,
                    "val_f1": 0.5,
                    "train_images_per_sec": 11.0,
                    "val_images_per_sec": 8.5,
                    "lr": 0.0005,
                },
            ]
            val_rows = [
                {"epoch": 1, "val_loss": 0.7, "val_f1": 0.3, "val_images_per_sec": 8.0},
                {"epoch": 2, "val_loss": 0.55, "val_f1": 0.5, "val_images_per_sec": 8.5},
            ]

            result = save_training_curves(target, train_rows, val_rows)

            self.assertEqual(result, target)
            self.assertTrue(target.exists())
            self.assertGreater(target.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
