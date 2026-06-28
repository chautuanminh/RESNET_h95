import pickle
import tempfile
import unittest
from pathlib import Path

from src.tamper_types import (
    assign_tamper_type,
    heuristic_tamper_type,
    load_tamper_metadata,
    normalize_tamper_label,
)


class TamperTypeTests(unittest.TestCase):
    def test_label_normalization(self):
        self.assertEqual(normalize_tamper_label("CM"), "copy_move")
        self.assertEqual(normalize_tamper_label("copy-move"), "copy_move")
        self.assertEqual(normalize_tamper_label("SP"), "splicing")
        self.assertEqual(normalize_tamper_label("generated"), "generation")

    def test_metadata_priority_over_heuristic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with (root / "DocTamperV1-TestingSet.pk").open("wb") as f:
                pickle.dump({7: "SP"}, f)
            metadata = load_tamper_metadata(root)
            assigned = assign_tamper_type(
                "TestingSet",
                7,
                metadata,
                row={"tamper_type": "generation"},
                enable_heuristic=True,
                heuristic_features={"mask_area_ratio": 0.5},
            )
            self.assertEqual(assigned.tamper_type, "splicing")
            self.assertEqual(assigned.tamper_type_source, "metadata")

    def test_unknown_when_no_metadata_and_heuristic_disabled(self):
        assigned = assign_tamper_type(
            "TestingSet",
            1,
            {},
            row={},
            enable_heuristic=False,
            heuristic_features={},
        )
        self.assertEqual(assigned.tamper_type, "unknown")
        self.assertEqual(assigned.tamper_type_source, "unknown")

    def test_heuristic_returns_valid_schema(self):
        result = heuristic_tamper_type(
            {
                "mask_area_ratio": 0.02,
                "component_count": 3,
                "patch_similarity": 0.92,
                "h95_inside_outside_ratio": 1.1,
            }
        )
        self.assertTrue(result.tamper_type.startswith("heuristic_"))
        self.assertGreaterEqual(result.tamper_type_confidence, 0.0)
        self.assertLessEqual(result.tamper_type_confidence, 1.0)
        self.assertTrue(result.tamper_type_reason)


if __name__ == "__main__":
    unittest.main()
