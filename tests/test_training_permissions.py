import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import load_config, resolve_output_paths
from src.train import TrainingLogger
from src.utils import ensure_writable_dir


class TrainingPermissionTests(unittest.TestCase):
    def test_ensure_writable_dir_creates_missing_nested_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "runs" / "tiny" / "nested"

            result = ensure_writable_dir(target, purpose="test output")

            self.assertEqual(result, target)
            self.assertTrue(target.is_dir())

    def test_ensure_writable_dir_rejects_unwritable_directory_with_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "readonly"
            target.mkdir()

            with patch("src.utils.os.access", return_value=False):
                with self.assertRaises(PermissionError) as ctx:
                    ensure_writable_dir(target, purpose="training output")

            message = str(ctx.exception)
            self.assertIn("training output directory is not writable", message)
            self.assertIn(str(target), message)
            self.assertIn("whoami", message)
            self.assertIn("chmod u+w", message)
            self.assertNotIn("sudo", message.lower())

    def test_training_logger_creates_parent_directory_and_writes_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "runs" / "tiny" / "training.log"

            logger = TrainingLogger(log_path)
            try:
                logger("hello permissions")
            finally:
                logger.close()

            text = log_path.read_text(encoding="utf-8")
            self.assertIn("hello permissions", text)

    def test_committed_tiny_config_uses_runs_root_instead_of_res(self):
        paths = resolve_output_paths(load_config("configs/test_tiny.yaml"))

        self.assertEqual(paths["root"], Path("runs") / "resnet34_h95_tiny")
        self.assertEqual(
            paths["run"],
            Path("runs") / "resnet34_h95_tiny" / "doctamper_resnet34_h95_35epochs_comparison",
        )
        self.assertNotEqual(paths["root"].parts[0], "res")


if __name__ == "__main__":
    unittest.main()
