from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import deep_get
from .utils import write_text


@dataclass
class DummyResNetUNet:
    in_channels: int = 2
    classes: int = 1
    first_conv_shape: list[int] | None = None

    def __post_init__(self) -> None:
        self.first_conv_shape = [64, self.in_channels, 7, 7]

    def __call__(self, x):
        try:
            import numpy as np  # type: ignore

            shape = list(x.shape)
            if len(shape) != 4:
                raise ValueError(f"Expected BCHW input, got {shape}")
            return np.zeros((shape[0], self.classes, shape[2], shape[3]), dtype=np.float32)
        except ImportError as exc:
            raise ImportError("Dummy model forward requires numpy") from exc


def create_model(config: dict[str, Any] | None = None):
    config = config or {}
    input_mode = deep_get(config, "model.input_mode", "gray_h95")
    if input_mode != "gray_h95":
        raise ValueError(f"Unsupported input_mode for ResNet34-H95: {input_mode}")
    in_channels = int(deep_get(config, "model.in_channels", 2))
    if in_channels != 2:
        raise ValueError(f"gray_h95 input requires exactly 2 input channels, got {in_channels}")
    classes = int(deep_get(config, "model.classes", 1))
    allow_dummy = bool(deep_get(config, "model.allow_dummy", False) or deep_get(config, "runtime.dummy", False))
    force_dummy = bool(deep_get(config, "model.force_dummy", False) or deep_get(config, "runtime.dummy", False))

    if not force_dummy:
        try:
            import segmentation_models_pytorch as smp  # type: ignore

            return smp.Unet(
                encoder_name=deep_get(config, "model.encoder_name", "resnet34"),
                encoder_weights=deep_get(config, "model.encoder_weights", "imagenet"),
                in_channels=in_channels,
                classes=classes,
                activation=None,
            )
        except Exception:
            if not allow_dummy:
                raise

    return DummyResNetUNet(in_channels=in_channels, classes=classes)


def get_first_conv_shape(model) -> list[int]:
    if hasattr(model, "first_conv_shape"):
        return list(model.first_conv_shape)
    candidates = [
        ("encoder", "conv1"),
        ("encoder", "_conv_stem"),
    ]
    for owner_name, conv_name in candidates:
        owner = getattr(model, owner_name, None)
        conv = getattr(owner, conv_name, None) if owner is not None else None
        weight = getattr(conv, "weight", None)
        shape = getattr(weight, "shape", None)
        if shape is not None:
            return [int(value) for value in shape]
    modules = getattr(model, "modules", None)
    if callable(modules):
        for module in modules():
            weight = getattr(module, "weight", None)
            shape = getattr(weight, "shape", None)
            if shape is not None and len(shape) == 4:
                return [int(value) for value in shape]
    raise ValueError("Unable to locate first convolution shape")


def forward_shape(model, x) -> list[int]:
    y = model(x)
    return [int(value) for value in y.shape]


def write_model_summary(path: str | Path, model, config: dict[str, Any]) -> Path:
    shape = get_first_conv_shape(model)
    lines = [
        "model name: ResNet34-H95 UNet",
        "method: grayscale + H95",
        f"configured model name: {deep_get(config, 'model.name', 'resnet34_unet')}",
        f"encoder name: {deep_get(config, 'model.encoder_name', 'resnet34')}",
        f"encoder weights: {deep_get(config, 'model.encoder_weights', 'imagenet')}",
        f"input mode: {deep_get(config, 'model.input_mode', 'gray_h95')}",
        f"input channels: {deep_get(config, 'model.in_channels', 2)}",
        f"output classes: {deep_get(config, 'model.classes', 1)}",
        f"first convolution weight shape: {shape}",
        f"first conv has exactly 2 input channels: {'yes' if shape[1] == 2 else 'no'}",
    ]
    return write_text(path, "\n".join(lines) + "\n")
