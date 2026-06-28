import tempfile
import unittest
import csv
from pathlib import Path

from src.run_all import run_all


class SmokeTests(unittest.TestCase):
    def test_run_all_tiny_synthetic_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "res"
            config = Path(tmp) / "tiny.yaml"
            config.write_text(
                f"""
runtime:
  dummy: true
data:
  root: synthetic
  tamper_metadata_dir: tampering_types
  train_sets: [DocTamperV1-TrainingSet]
  test_sets:
    TestingSet: DocTamperV1-TestingSet
    FCD: DocTamperV1-FCD
    SCD: DocTamperV1-SCD
  synthetic:
    enabled: true
    default_count: 2
    counts:
      DocTamperV1-TrainingSet: 8
      TrainingSet: 8
      DocTamperV1-TestingSet: 2
      TestingSet: 2
      DocTamperV1-FCD: 2
      FCD: 2
      DocTamperV1-SCD: 2
      SCD: 2
split:
  seed: 42
  val_count: 2
preprocessing:
  image_size: 32
model:
  in_channels: 2
  classes: 1
  allow_dummy: true
  force_dummy: true
training:
  epochs: 1
  batch_size: 2
evaluation:
  threshold_min: 0.01
  threshold_max: 0.03
  threshold_step: 0.01
  official_threshold: 0.5
  batch_size: 2
  save_examples_per_set: 1
failure_analysis:
  top_k: 1
  output_dir: failure_case_analysis
tampering_type_analysis:
  enabled: true
  output_dir: tampering_type_analysis
  threshold_min: 0.01
  threshold_max: 0.03
  threshold_step: 0.01
  official_threshold: 0.5
  worst_k_per_type: 1
  save_examples_per_type: 1
  use_metadata_if_available: true
  enable_heuristic_classifier: true
output:
  root_dir: {root.as_posix()}
  run_dir: doctamper_resnet34_h95_35epochs_comparison
validation:
  require_outputs: true
""",
                encoding="utf-8",
            )
            run_all(config)
            self.assertTrue((root / "validate_experiment_report.md").exists())
            self.assertTrue((root / "tampering_type_analysis" / "tampering_type_per_image.csv").exists())
            run_dir = root / "doctamper_resnet34_h95_35epochs_comparison"
            self.assertTrue((run_dir / "config_resolved.yaml").exists())
            self.assertTrue((run_dir / "training.log").exists())
            self.assertTrue((run_dir / "plots" / "training_curves.png").exists())
            with (run_dir / "train_metrics.csv").open(newline="", encoding="utf-8") as f:
                headers = set(next(csv.reader(f)))
            self.assertTrue(
                {
                    "epoch",
                    "train_loss",
                    "val_loss",
                    "train_f1",
                    "val_f1",
                    "train_images_per_sec",
                    "val_images_per_sec",
                    "gpu_allocated_gb",
                    "gpu_reserved_gb",
                    "gpu_max_reserved_gb",
                    "gpu_max_reserved_percent",
                    "batch_size",
                    "effective_batch_size",
                    "amp_dtype",
                }.issubset(headers)
            )


if __name__ == "__main__":
    unittest.main()
