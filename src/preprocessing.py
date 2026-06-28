from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any


@dataclass(frozen=True)
class PreprocessedSample:
    input: Any
    mask: Any
    gray: Any
    h95: Any


def _require_numpy_pillow():
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise ImportError("preprocessing requires numpy and pillow") from exc
    return np, Image


def _to_gray_image(image: Any):
    np, Image = _require_numpy_pillow()
    if isinstance(image, Image.Image):
        return image.convert("L")
    array = np.asarray(image)
    if array.ndim == 2:
        return Image.fromarray(array.astype("uint8"), mode="L")
    return Image.fromarray(array.astype("uint8")).convert("L")


def compute_h95_residual(image: Any, jpeg_quality: int = 95, eps: float = 1e-8):
    np, Image = _require_numpy_pillow()
    gray_image = _to_gray_image(image)
    original = np.asarray(gray_image, dtype=np.uint8)

    buffer = BytesIO()
    gray_image.save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("L")
    jpeg_gray = np.asarray(recompressed, dtype=np.uint8)

    diff = np.abs(original.astype("float32") - jpeg_gray.astype("float32"))
    p99 = float(np.percentile(diff, 99))
    h95 = np.clip(diff / (p99 + eps), 0.0, 1.0).astype("float32")
    return h95


def compute_h95_before_resize(image: Any, jpeg_quality: int = 95, eps: float = 1e-8):
    return compute_h95_residual(image, jpeg_quality=jpeg_quality, eps=eps)


def preprocess_gray_h95(
    image: Any,
    mask: Any | None,
    size: tuple[int, int] = (512, 512),
    jpeg_quality: int = 95,
) -> PreprocessedSample:
    np, Image = _require_numpy_pillow()
    gray_image = _to_gray_image(image)

    width, height = size
    gray_resized = gray_image.resize((width, height), Image.Resampling.BILINEAR)
    gray_uint8 = np.asarray(gray_resized, dtype=np.uint8)
    h95_resized = compute_h95_residual(Image.fromarray(gray_uint8, mode="L"), jpeg_quality=jpeg_quality)

    gray = (gray_uint8.astype(np.float32) / 255.0).astype("float32")
    h95_resized = np.clip(h95_resized, 0.0, 1.0).astype("float32")

    if mask is None:
        mask_array = np.zeros((height, width), dtype=np.float32)
    else:
        if isinstance(mask, Image.Image):
            mask_image = mask.convert("L")
        else:
            mask_image = Image.fromarray(np.asarray(mask).astype("uint8")).convert("L")
        mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
        mask_array = (np.asarray(mask_image, dtype=np.uint8) > 0).astype("float32")

    stacked = np.stack([gray, h95_resized], axis=0).astype("float32")
    return PreprocessedSample(
        input=stacked,
        mask=mask_array[None, :, :].astype("float32"),
        gray=gray,
        h95=h95_resized,
    )
