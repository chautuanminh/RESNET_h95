import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.smoke import main


class SmokeCliTests(unittest.TestCase):
    def test_smoke_cli_reports_dataset_and_sample_shapes_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
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
  synthetic:
    enabled: true
    default_count: 3
    counts:
      DocTamperV1-TrainingSet: 3
      DocTamperV1-TestingSet: 2
split:
  seed: 42
  val_count: 1
preprocessing:
  image_size: 16
model:
  in_channels: 2
  classes: 1
  allow_dummy: true
  force_dummy: true
output:
  root_dir: {(Path(tmp) / "runs").as_posix()}
  run_dir: smoke
""",
                encoding="utf-8",
            )
            output = io.StringIO()

            with redirect_stdout(output):
                result = main(["--config", str(config), "--sample-count", "2"])

            text = output.getvalue()
            self.assertEqual(result, 0)
            self.assertIn("cwd:", text)
            self.assertIn("runtime:", text)
            self.assertIn("output.run_dir:", text)
            self.assertIn("DocTamperV1-TrainingSet", text)
            self.assertIn("input_shape=(2, 16, 16)", text)
            self.assertIn("mask_shape=(1, 16, 16)", text)


if __name__ == "__main__":
    unittest.main()
