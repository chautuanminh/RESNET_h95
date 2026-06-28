import tempfile
import unittest
import math
from pathlib import Path

from src.config import load_config, resolve_output_paths
from src.evaluate import run_evaluate
from src.failure_analysis import (
    choose_primary_failure_category,
    choose_raw_pixel_category,
    compute_h95_diagnostics,
    compute_threshold_sweep_rows,
    connected_component_stats,
    select_worst_k,
    severity_from_f1,
)
from src.train import run_train
from src.utils import read_csv_rows
from src.validate_experiment import validate_required_outputs, validate_no_forbidden_artifacts


class FailureAndValidationTests(unittest.TestCase):
    def test_failure_selection_is_stable(self):
        rows = [
            {"dataset_index": 3, "f1": 0.2, "iou": 0.3},
            {"dataset_index": 2, "f1": 0.1, "iou": 0.4},
            {"dataset_index": 1, "f1": 0.1, "iou": 0.2},
        ]
        selected = select_worst_k(rows, 2)
        self.assertEqual([row["dataset_index"] for row in selected], [1, 2])

    def test_severity_uses_expected_f1_bands(self):
        self.assertEqual(severity_from_f1(0.249), "catastrophic")
        self.assertEqual(severity_from_f1(0.25), "severe")
        self.assertEqual(severity_from_f1(0.5), "moderate")
        self.assertEqual(severity_from_f1(0.75), "minor")

    def test_raw_pixel_category_uses_strict_two_x_heavy_rule(self):
        self.assertEqual(choose_raw_pixel_category(0, 0, 0, 0, 0), "perfect")
        self.assertEqual(choose_raw_pixel_category(0, 3, 0, 0, 3), "hallucination_no_gt")
        self.assertEqual(choose_raw_pixel_category(0, 0, 4, 4, 0), "missed_all_tamper")
        self.assertEqual(choose_raw_pixel_category(5, 9, 5, 10, 14), "mixed_fp_fn")
        self.assertEqual(choose_raw_pixel_category(5, 10, 5, 10, 15), "false_positive_heavy")
        self.assertEqual(choose_raw_pixel_category(5, 5, 10, 15, 10), "false_negative_heavy")

    def test_primary_failure_category_priority_order(self):
        base = {
            "f1": 0.2,
            "iou": 0.1,
            "precision": 0.8,
            "recall": 0.4,
            "pred_gt_area_ratio": 3.0,
            "threshold_gap_f1": 0.3,
            "pred_num_blobs": 20,
            "gt_num_blobs": 1,
        }
        self.assertEqual(choose_primary_failure_category(base), "false_negative_missed_tamper")
        over_expansion = dict(base, f1=0.7, iou=0.55, precision=0.8, recall=0.8)
        self.assertEqual(choose_primary_failure_category(over_expansion), "over_expansion")
        minor = dict(base, f1=0.9, iou=0.8, precision=0.1, recall=0.1)
        self.assertEqual(choose_primary_failure_category(minor), "good_or_minor_error")

    def test_connected_component_stats_use_four_connectivity(self):
        import numpy as np

        mask = np.array([[1, 0], [0, 1]], dtype=bool)
        stats = connected_component_stats(mask)
        self.assertEqual(stats["num_blobs"], 2)
        self.assertEqual(stats["largest_blob_area"], 1)

    def test_threshold_sweep_row_count_and_best_tie_break(self):
        import numpy as np

        prob = np.array([[0.9, 0.9], [0.1, 0.1]], dtype="float32")
        gt = np.array([[1, 1], [0, 0]], dtype="float32")
        rows, best = compute_threshold_sweep_rows(
            prob,
            gt,
            [0.2, 0.4],
            {"test_set": "T", "dataset_index": 7},
        )
        self.assertEqual(len(rows), 2)
        self.assertEqual(best["best_threshold"], 0.2)
        self.assertEqual(best["best_threshold_f1"], 1.0)

    def test_h95_unavailable_outputs_nan_diagnostics(self):
        import numpy as np

        gt = np.array([[1, 0], [0, 0]], dtype=bool)
        pred = np.array([[0, 1], [0, 0]], dtype=bool)
        diagnostics = compute_h95_diagnostics(None, gt, pred, h95_available=False)
        self.assertFalse(diagnostics["h95_available"])
        self.assertTrue(math.isnan(diagnostics["h95_mean_inside_gt"]))
        self.assertEqual(diagnostics["likely_h95_signal"], "not_available_no_h95_model")

    def test_failure_analysis_generates_rich_artifacts_for_tiny_synthetic_pipeline(self):
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
    default_count: 3
    counts:
      DocTamperV1-TrainingSet: 8
      TrainingSet: 8
      DocTamperV1-TestingSet: 3
      TestingSet: 3
      DocTamperV1-FCD: 3
      FCD: 3
      DocTamperV1-SCD: 3
      SCD: 3
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
  top_k: 2
  output_dir: failure_case_analysis
tampering_type_analysis:
  enabled: false
output:
  root_dir: {root.as_posix()}
  run_dir: doctamper_resnet34_h95_35epochs_comparison
validation:
  require_outputs: false
""",
                encoding="utf-8",
            )
            checkpoint = run_train(config_path)
            run_evaluate(config_path, str(checkpoint))

            from src.failure_analysis import run_failure_analysis

            run_failure_analysis(config_path, str(checkpoint))
            failure_dir = resolve_output_paths(load_config(config_path))["failure"]
            selected = read_csv_rows(failure_dir / "failure_cases_all_selected.csv")
            sweep = read_csv_rows(failure_dir / "threshold_sweep_0.01_0.99.csv")
            self.assertEqual(len(selected), 6)
            self.assertEqual(len(sweep), 18)

            for name in [
                "failure_analysis_config.yaml",
                "failure_analysis_config.json",
                "failure_summary_by_test_set.csv",
                "failure_summary_by_primary_category.csv",
                "failure_summary_by_raw_category.csv",
                "failure_summary_by_severity.csv",
                "failure_summary_by_likely_reason.csv",
                "failure_summary_by_category.csv",
                "failure_summary_by_likely_tamper_type.csv",
                "failure_case_report.md",
            ]:
                self.assertTrue((failure_dir / name).exists(), name)

            for name in [
                "worst200_f1_distribution.png",
                "failure_category_counts.png",
                "severity_counts_by_test_set.png",
                "raw_category_counts.png",
                "precision_recall_scatter.png",
                "threshold_gap_distribution.png",
                "pred_gt_area_ratio_vs_f1.png",
                "h95_signal_ratio_vs_f1.png",
                "confusion_totals_by_test_set.png",
            ]:
                self.assertTrue((failure_dir / "plots" / name).exists(), name)

            panel_path = Path(selected[0]["panel_path"])
            self.assertTrue(panel_path.exists())
            from PIL import Image

            with Image.open(panel_path) as image:
                self.assertEqual(image.size[0] * 2, image.size[1] * 5)

    def test_output_validation_fails_when_required_files_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ok, errors = validate_required_outputs(Path(tmp), require_tamper_outputs=True)
            self.assertFalse(ok)
            self.assertTrue(errors)

    def test_forbidden_artifact_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ("bad_" + "blob" + "_minarea.csv")).write_text("x", encoding="utf-8")
            ok, errors = validate_no_forbidden_artifacts(root)
            self.assertFalse(ok)
            self.assertTrue(errors)


if __name__ == "__main__":
    unittest.main()
