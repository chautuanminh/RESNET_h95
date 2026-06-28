import importlib.util
import unittest


HAS_NUMPY = importlib.util.find_spec("numpy") is not None
HAS_PIL = importlib.util.find_spec("PIL") is not None

if HAS_NUMPY and HAS_PIL:
    import numpy as np
    from PIL import Image

    from src.preprocessing import compute_h95_residual, preprocess_gray_h95


@unittest.skipUnless(HAS_NUMPY and HAS_PIL, "numpy and pillow are required")
class PreprocessingTests(unittest.TestCase):
    def test_h95_range_shape_and_determinism(self):
        image = np.tile(np.arange(64, dtype=np.uint8), (48, 1))
        mask = np.zeros((48, 64), dtype=np.uint8)
        mask[10:20, 11:25] = 255

        first = preprocess_gray_h95(image, mask, size=(32, 32))
        second = preprocess_gray_h95(image, mask, size=(32, 32))

        self.assertEqual(first.input.shape, (2, 32, 32))
        self.assertEqual(first.mask.shape, (1, 32, 32))
        self.assertGreaterEqual(float(first.h95.min()), 0.0)
        self.assertLessEqual(float(first.h95.max()), 1.0)
        self.assertTrue(np.array_equal(first.h95, second.h95))
        self.assertFalse(np.isnan(first.h95).any())
        self.assertFalse(np.isinf(first.h95).any())

    def test_h95_is_computed_after_resizing_uint8_image(self):
        y_grid, x_grid = np.mgrid[0:48, 0:64]
        image = ((x_grid * 7 + y_grid * 11 + (x_grid * y_grid) % 37) % 256).astype("uint8")

        out = preprocess_gray_h95(image, None, size=(32, 32))
        resized = Image.fromarray(image, mode="L").resize((32, 32), Image.Resampling.BILINEAR)
        expected_h95 = compute_h95_residual(resized, jpeg_quality=95)

        self.assertTrue(np.allclose(out.h95, expected_h95, atol=1e-6))

    def test_mask_is_not_leaked_into_input_channels(self):
        image = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        empty_mask = np.zeros((64, 64), dtype=np.uint8)
        full_mask = np.full((64, 64), 255, dtype=np.uint8)

        empty = preprocess_gray_h95(image, empty_mask, size=(32, 32))
        full = preprocess_gray_h95(image, full_mask, size=(32, 32))

        self.assertTrue(np.array_equal(empty.input, full.input))
        self.assertFalse(np.array_equal(empty.mask, full.mask))

    def test_zero_p99_h95_has_no_nan_or_inf(self):
        image = np.zeros((20, 20), dtype=np.uint8)
        out = preprocess_gray_h95(image, None, size=(16, 16))
        self.assertFalse(np.isnan(out.h95).any())
        self.assertFalse(np.isinf(out.h95).any())
        self.assertTrue((out.h95 >= 0).all())
        self.assertTrue((out.h95 <= 1).all())

    def test_mask_uses_nearest_neighbor_values(self):
        image = np.zeros((4, 4), dtype=np.uint8)
        mask = np.array(
            [
                [0, 0, 255, 255],
                [0, 0, 255, 255],
                [255, 255, 0, 0],
                [255, 255, 0, 0],
            ],
            dtype=np.uint8,
        )
        out = preprocess_gray_h95(image, mask, size=(8, 8))
        self.assertEqual(set(np.unique(out.mask).tolist()), {0.0, 1.0})


if __name__ == "__main__":
    unittest.main()
