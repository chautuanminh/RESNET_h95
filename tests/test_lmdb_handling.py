import pickle
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.datasets import DocTamperLMDBDataset, discover_indices


try:
    import lmdb  # type: ignore
except Exception:
    lmdb = None


def _png_bytes(value: int) -> bytes:
    image = Image.new("L", (8, 8), value)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@unittest.skipIf(lmdb is None, "lmdb package is not installed")
class LMDBHandlingTests(unittest.TestCase):
    def test_discovery_and_dataset_read_work_while_lmdb_env_is_already_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            folder = "CustomSet"
            path = root / folder
            path.mkdir(parents=True)
            metadata_root = Path(tmp) / "metadata"
            metadata_root.mkdir()
            env = lmdb.open(str(path), map_size=10_485_760, max_readers=8)
            try:
                with env.begin(write=True) as txn:
                    txn.put(b"num-samples", b"2")
                    txn.put(
                        b"sample_000000",
                        pickle.dumps({"image": _png_bytes(160), "mask": _png_bytes(255)}),
                    )
                    txn.put(
                        b"sample_000001",
                        pickle.dumps({"image": _png_bytes(170), "mask": _png_bytes(0)}),
                    )
                config = {
                    "data": {
                        "root": str(root),
                        "tamper_metadata_dir": str(metadata_root),
                        "synthetic": {"enabled": False},
                    },
                    "preprocessing": {"image_size": 8},
                }

                self.assertEqual(discover_indices(config, folder), [0, 1])
                dataset = DocTamperLMDBDataset(root, "custom", folder, [0], image_size=8, metadata_root=metadata_root)
                try:
                    sample = dataset[0]
                    self.assertEqual(sample["dataset_index"], 0)
                    self.assertEqual(sample["input"].shape, (2, 8, 8))
                    self.assertEqual(sample["mask"].shape, (1, 8, 8))
                finally:
                    dataset.close()
            finally:
                env.close()

    def test_dataset_pickle_drops_open_lmdb_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            folder = "CustomSet"
            path = root / folder
            path.mkdir(parents=True)
            metadata_root = Path(tmp) / "metadata"
            metadata_root.mkdir()
            env = lmdb.open(str(path), map_size=10_485_760, max_readers=8)
            try:
                with env.begin(write=True) as txn:
                    txn.put(b"sample_000000", pickle.dumps({"image": _png_bytes(160), "mask": _png_bytes(255)}))
                dataset = DocTamperLMDBDataset(root, "custom", folder, [0], image_size=8, metadata_root=metadata_root)
                dataset[0]
                self.assertIsNotNone(dataset._env)

                restored = pickle.loads(pickle.dumps(dataset))

                self.assertIsNone(restored._env)
                restored.close()
                dataset.close()
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
