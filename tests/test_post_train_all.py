import tempfile
import unittest
from pathlib import Path

from src.config import load_config, resolve_output_paths
from src.post_train_all import run_post_train_all
from src.train import run_train


class PostTrainAllTests(unittest.TestCase):
    def test_post_train_all_runs_without_retraining(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "res"
            config_path = Path(tmp) / "tiny.yaml"
            config_path.write_text(
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
            checkpoint = run_train(config_path)
            train_metrics_mtime = (
                resolve_output_paths(load_config(config_path))["run"] / "train_metrics.csv"
            ).stat().st_mtime
            run_post_train_all(config_path, str(checkpoint))
            self.assertEqual(
                train_metrics_mtime,
                (resolve_output_paths(load_config(config_path))["run"] / "train_metrics.csv").stat().st_mtime,
            )
            self.assertTrue((root / "validate_experiment_report.md").exists())


if __name__ == "__main__":
    unittest.main()
