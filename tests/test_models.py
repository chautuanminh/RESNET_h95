import importlib.util
import unittest


HAS_NUMPY = importlib.util.find_spec("numpy") is not None

if HAS_NUMPY:
    import numpy as np

from src.models import create_model, get_first_conv_shape, forward_shape


class ModelTests(unittest.TestCase):
    def test_model_contract_supports_two_channel_input(self):
        model = create_model({"model": {"allow_dummy": True, "in_channels": 2}})
        self.assertEqual(get_first_conv_shape(model), [64, 2, 7, 7])

    def test_model_metadata_defaults_to_gray_h95_contract(self):
        config = {
            "model": {
                "name": "resnet34_unet",
                "input_mode": "gray_h95",
                "allow_dummy": True,
                "force_dummy": True,
            }
        }
        model = create_model(config)
        self.assertEqual(get_first_conv_shape(model)[1], 2)

    @unittest.skipUnless(HAS_NUMPY, "numpy is required for dummy forward")
    def test_forward_shape(self):
        model = create_model({"model": {"allow_dummy": True, "in_channels": 2}})
        x = np.zeros((2, 2, 512, 512), dtype=np.float32)
        self.assertEqual(forward_shape(model, x), [2, 1, 512, 512])


if __name__ == "__main__":
    unittest.main()
