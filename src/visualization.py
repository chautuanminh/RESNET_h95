from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_text


def save_diagnostic_panel(
    path: str | Path,
    sample: dict[str, Any],
    probability,
    prediction,
    metrics: dict[str, Any],
    title: str = "ResNet34-H95 diagnostic panel",
) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    try:
        import numpy as np  # type: ignore
        from PIL import Image, ImageDraw  # type: ignore

        gray = _to_uint8(sample.get("gray"))
        h95 = _to_uint8(sample.get("h95"))
        gt = _to_uint8(sample.get("mask")[0])
        prob = _to_uint8(probability)
        pred = _to_uint8(prediction)
        tp = ((prediction > 0) & (sample.get("mask")[0] > 0.5)).astype("uint8") * 255
        fp = ((prediction > 0) & (sample.get("mask")[0] <= 0.5)).astype("uint8") * 255
        fn = ((prediction <= 0) & (sample.get("mask")[0] > 0.5)).astype("uint8") * 255
        error = np.zeros((*gt.shape, 3), dtype="uint8")
        error[..., 1] = tp
        error[..., 0] = fp
        error[..., 2] = fn
        overlay = np.stack([gray, gray, gray], axis=-1)
        overlay[..., 0] = np.maximum(overlay[..., 0], fp)
        overlay[..., 1] = np.maximum(overlay[..., 1], tp)
        overlay[..., 2] = np.maximum(overlay[..., 2], fn)
        tiles = [
            ("gray", np.stack([gray, gray, gray], axis=-1)),
            ("H95", np.stack([h95, h95, h95], axis=-1)),
            ("ground truth", np.stack([gt, gt, gt], axis=-1)),
            ("probability", np.stack([prob, prob, prob], axis=-1)),
            ("prediction 0.5", np.stack([pred, pred, pred], axis=-1)),
            ("TP/FP/FN", error),
            ("overlay", overlay),
        ]
        tile_h, tile_w = gray.shape
        text_h = tile_h
        canvas = Image.new("RGB", (tile_w * 4, tile_h * 2), "white")
        draw = ImageDraw.Draw(canvas)
        for i, (label, arr) in enumerate(tiles):
            x = (i % 4) * tile_w
            y = (i // 4) * tile_h
            canvas.paste(Image.fromarray(arr).resize((tile_w, tile_h)), (x, y))
            draw.rectangle([x, y, x + min(190, tile_w), y + 15], fill="white")
            draw.text((x + 4, y + 2), label, fill="black")
        x = 3 * tile_w
        y = tile_h
        draw.rectangle([x, y, x + tile_w, y + text_h], fill="white")
        lines = [
            title,
            f"set: {sample.get('test_set')}",
            f"index: {sample.get('dataset_index')}",
            f"type: {sample.get('tamper_type')}",
            f"source: {sample.get('tamper_type_source')}",
            f"confidence: {sample.get('tamper_type_confidence')}",
            f"F1: {float(metrics.get('f1', 0.0)):.4f}",
            f"IoU: {float(metrics.get('iou', 0.0)):.4f}",
            f"P/R: {float(metrics.get('precision', 0.0)):.3f}/{float(metrics.get('recall', 0.0)):.3f}",
            f"reason: {str(sample.get('tamper_type_reason', ''))[:42]}",
        ]
        for offset, line in enumerate(lines):
            draw.text((x + 6, y + 8 + offset * 14), line, fill="black")
        canvas.save(target)
    except Exception:
        write_text(
            target,
            "\n".join(
                [
                    title,
                    f"test_set: {sample.get('test_set')}",
                    f"dataset_index: {sample.get('dataset_index')}",
                    f"tamper_type: {sample.get('tamper_type')}",
                    f"tamper_type_source: {sample.get('tamper_type_source')}",
                    f"metrics: {metrics}",
                ]
            ),
        )
    return target


def save_failure_diagnostic_panel(
    path: str | Path,
    sample: dict[str, Any],
    probability,
    prediction,
    diagnostics: dict[str, Any],
    h95_available: bool = True,
) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    try:
        import numpy as np  # type: ignore
        from PIL import Image, ImageDraw  # type: ignore

        tile = 220
        gray = _to_uint8(sample.get("gray"))
        gt_mask = (np.asarray(sample.get("mask")[0]) > 0.5)
        pred_mask = np.asarray(prediction).astype(bool)
        prob = np.asarray(probability, dtype="float32")
        tp = pred_mask & gt_mask
        fp = pred_mask & ~gt_mask
        fn = ~pred_mask & gt_mask

        rgb = np.stack([gray, gray, gray], axis=-1)
        h95 = _heatmap(sample.get("h95")) if h95_available and sample.get("h95") is not None else np.zeros_like(rgb)
        gt = _mask_rgb(gt_mask, (255, 255, 255))
        prob_tile = _heatmap(prob)
        pred = _mask_rgb(pred_mask, (255, 255, 255))
        tp_tile = _mask_rgb(tp, (0, 180, 90))
        fp_tile = _mask_rgb(fp, (220, 70, 45))
        fn_tile = _mask_rgb(fn, (40, 105, 210))
        overlay = rgb.copy()
        overlay = _blend_mask(overlay, gt_mask, (0, 190, 80), 0.45)
        overlay = _blend_mask(overlay, pred_mask, (230, 60, 45), 0.35)

        tiles = [
            ("RGB image", rgb),
            ("H95 heatmap" if h95_available else "H95 N/A", h95),
            ("Ground truth", gt),
            ("Probability", prob_tile),
            ("Prediction 0.5", pred),
            ("TP", tp_tile),
            ("FP", fp_tile),
            ("FN", fn_tile),
            ("GT + prediction", overlay),
        ]
        canvas = Image.new("RGB", (tile * 5, tile * 2), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (label, arr) in enumerate(tiles):
            x = (index % 5) * tile
            y = (index // 5) * tile
            image = Image.fromarray(arr.astype("uint8")).resize((tile, tile))
            canvas.paste(image, (x, y))
            draw.rectangle([x, y, x + tile, y + 18], fill="white")
            draw.text((x + 5, y + 3), label, fill="black")

        text_x = 4 * tile
        text_y = tile
        draw.rectangle([text_x, text_y, text_x + tile, text_y + tile], fill="white")
        text_lines = _diagnostic_text_lines(diagnostics)
        for offset, line in enumerate(text_lines[:15]):
            draw.text((text_x + 7, text_y + 7 + offset * 14), line, fill="black")
        canvas.save(target)
    except Exception:
        write_text(
            target,
            "\n".join(
                [
                    "Failure diagnostic panel",
                    f"test_set: {sample.get('test_set')}",
                    f"dataset_index: {sample.get('dataset_index')}",
                    f"diagnostics: {diagnostics}",
                ]
            ),
        )
    return target


def save_placeholder_plot(path: str | Path, title: str) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    try:
        from PIL import Image, ImageDraw  # type: ignore

        image = Image.new("RGB", (800, 450), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 24), title, fill="black")
        image.save(target)
    except Exception:
        write_text(target, title)
    return target


def save_training_curves(path: str | Path, train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]] | None = None) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    val_rows = val_rows or []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        _plot_metric_pair(axes[0][0], train_rows, val_rows, "train_loss", "val_loss", "Loss", "loss")
        _plot_metric_pair(axes[0][1], train_rows, val_rows, "train_f1", "val_f1", "F1", "f1")
        _plot_metric_pair(
            axes[1][0],
            train_rows,
            val_rows,
            "train_images_per_sec",
            "val_images_per_sec",
            "Throughput",
            "images/sec",
        )
        _plot_single_metric(axes[1][1], train_rows, "lr", "Learning rate", "lr")
        fig.suptitle("Training curves")
        fig.tight_layout()
        fig.savefig(target, dpi=150)
        plt.close(fig)
    except Exception:
        _save_training_curves_fallback(target, train_rows, val_rows)
    return target


def _to_uint8(array):
    import numpy as np  # type: ignore

    arr = np.asarray(array)
    if arr.dtype == np.bool_:
        arr = arr.astype("float32")
    if arr.max(initial=0) <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype("uint8")


def _heatmap(array):
    import numpy as np  # type: ignore

    value = np.asarray(array, dtype="float32")
    if value.size == 0:
        value = np.zeros((1, 1), dtype="float32")
    value = value - np.nanmin(value)
    max_value = float(np.nanmax(value)) if value.size else 0.0
    if max_value > 0:
        value = value / max_value
    value = np.nan_to_num(value, nan=0.0)
    red = np.clip(255 * value, 0, 255)
    green = np.clip(255 * (1.0 - np.abs(value - 0.5) * 2.0), 0, 255)
    blue = np.clip(255 * (1.0 - value), 0, 255)
    return np.stack([red, green, blue], axis=-1).astype("uint8")


def _mask_rgb(mask, color: tuple[int, int, int]):
    import numpy as np  # type: ignore

    mask_array = np.asarray(mask).astype(bool)
    out = np.zeros((*mask_array.shape, 3), dtype="uint8")
    for channel, value in enumerate(color):
        out[..., channel] = mask_array.astype("uint8") * value
    return out


def _blend_mask(image, mask, color: tuple[int, int, int], alpha: float):
    import numpy as np  # type: ignore

    out = np.asarray(image, dtype="float32").copy()
    mask_array = np.asarray(mask).astype(bool)
    color_array = np.asarray(color, dtype="float32")
    out[mask_array] = out[mask_array] * (1.0 - alpha) + color_array * alpha
    return np.clip(out, 0, 255).astype("uint8")


def _diagnostic_text_lines(diagnostics: dict[str, Any]) -> list[str]:
    return [
        "Diagnostics",
        f"set/rank: {diagnostics.get('test_set')} #{diagnostics.get('rank_within_test_set')}",
        f"id: {diagnostics.get('image_id')}",
        f"idx: {diagnostics.get('dataset_index')}",
        f"F1/IoU: {_fmt(diagnostics.get('f1'))}/{_fmt(diagnostics.get('iou'))}",
        f"P/R: {_fmt(diagnostics.get('precision'))}/{_fmt(diagnostics.get('recall'))}",
        f"severity: {diagnostics.get('severity')}",
        f"primary: {diagnostics.get('primary_failure_category')}",
        f"raw: {diagnostics.get('raw_pixel_category')}",
        f"reason: {str(diagnostics.get('likely_reasons', ''))[:28]}",
        f"best t: {_fmt(diagnostics.get('best_threshold'))}",
        f"gap F1: {_fmt(diagnostics.get('threshold_gap_f1'))}",
        f"area pred/gt: {_fmt(diagnostics.get('pred_gt_area_ratio'))}",
        f"areas G/P/TP/FP/FN: {diagnostics.get('gt_area')}/{diagnostics.get('pred_area')}/{diagnostics.get('tp')}/{diagnostics.get('fp')}/{diagnostics.get('fn')}",
        f"blobs G/P: {diagnostics.get('gt_num_blobs')}/{diagnostics.get('pred_num_blobs')}",
        f"H95 ratio: {_fmt(diagnostics.get('h95_signal_ratio'))}",
    ]


def _fmt(value: Any) -> str:
    try:
        numeric = float(value)
        if numeric != numeric:
            return "nan"
        return f"{numeric:.3f}"
    except (TypeError, ValueError):
        return str(value)


def _metric_series(rows: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for ordinal, row in enumerate(rows, start=1):
        if key not in row:
            continue
        try:
            y = float(row[key])
            if y != y:
                continue
            x = float(row.get("epoch", ordinal))
        except (TypeError, ValueError):
            continue
        x_values.append(x)
        y_values.append(y)
    return x_values, y_values


def _plot_metric_pair(
    axis,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    train_key: str,
    val_key: str,
    title: str,
    ylabel: str,
) -> None:
    plotted = False
    train_x, train_y = _metric_series(train_rows, train_key)
    if train_y:
        axis.plot(train_x, train_y, marker="o", label=train_key)
        plotted = True
    val_source = val_rows if val_rows else train_rows
    val_x, val_y = _metric_series(val_source, val_key)
    if val_y:
        axis.plot(val_x, val_y, marker="o", label=val_key)
        plotted = True
    _finish_axis(axis, title, ylabel, plotted)


def _plot_single_metric(axis, rows: list[dict[str, Any]], key: str, title: str, ylabel: str) -> None:
    x_values, y_values = _metric_series(rows, key)
    plotted = bool(y_values)
    if plotted:
        axis.plot(x_values, y_values, marker="o", label=key)
    _finish_axis(axis, title, ylabel, plotted)


def _finish_axis(axis, title: str, ylabel: str, plotted: bool) -> None:
    axis.set_title(title)
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    if plotted:
        axis.legend()
    else:
        axis.text(0.5, 0.5, "no data", ha="center", va="center", transform=axis.transAxes)


def _save_training_curves_fallback(
    target: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> None:
    try:
        from PIL import Image, ImageDraw  # type: ignore

        image = Image.new("RGB", (900, 560), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 24), "Training curves", fill="black")
        latest_train = train_rows[-1] if train_rows else {}
        latest_val = val_rows[-1] if val_rows else {}
        lines = [
            f"epochs: {len(train_rows)}",
            f"latest train_loss: {latest_train.get('train_loss', 'n/a')}",
            f"latest val_loss: {latest_val.get('val_loss', latest_train.get('val_loss', 'n/a'))}",
            f"latest train_f1: {latest_train.get('train_f1', 'n/a')}",
            f"latest val_f1: {latest_val.get('val_f1', latest_train.get('val_f1', 'n/a'))}",
            f"latest train_images_per_sec: {latest_train.get('train_images_per_sec', 'n/a')}",
            f"latest val_images_per_sec: {latest_val.get('val_images_per_sec', latest_train.get('val_images_per_sec', 'n/a'))}",
            f"latest lr: {latest_train.get('lr', 'n/a')}",
        ]
        for offset, line in enumerate(lines):
            draw.text((24, 64 + offset * 24), line, fill="black")
        image.save(target)
    except Exception:
        write_text(target, "Training curves\n")
