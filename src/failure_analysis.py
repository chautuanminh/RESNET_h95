from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import deep_get, dump_config, load_config, resolve_output_paths
from .datasets import iter_batches, make_dataset
from .evaluate import _load_checkpoint_if_available
from .gpu import resolve_runtime
from .metrics import metrics_from_counts
from .models import create_model
from .utils import as_float, as_int, ensure_dir, frange, read_csv_rows, write_csv, write_json, write_text
from .visualization import save_failure_diagnostic_panel


SEVERITY_LEVELS = ["catastrophic", "severe", "moderate", "minor"]
FAILURE_PLOT_NAMES = [
    "worst200_f1_distribution.png",
    "failure_category_counts.png",
    "severity_counts_by_test_set.png",
    "raw_category_counts.png",
    "precision_recall_scatter.png",
    "threshold_gap_distribution.png",
    "pred_gt_area_ratio_vs_f1.png",
    "h95_signal_ratio_vs_f1.png",
    "confusion_totals_by_test_set.png",
]


def select_worst_k(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            as_float(row.get("f1"), 1.0),
            as_float(row.get("iou"), 1.0),
            as_int(row.get("dataset_index"), 0),
        ),
    )[:k]


def severity_from_f1(f1: float) -> str:
    value = float(f1)
    if value < 0.25:
        return "catastrophic"
    if value < 0.50:
        return "severe"
    if value < 0.75:
        return "moderate"
    return "minor"


def choose_raw_pixel_category(tp: int, fp: int, fn: int, gt_pos: int, pred_pos: int) -> str:
    if fp + fn == 0:
        return "perfect"
    if gt_pos == 0 and pred_pos > 0:
        return "hallucination_no_gt"
    if gt_pos > 0 and pred_pos == 0:
        return "missed_all_tamper"
    if fn == 0 and fp > 0:
        return "false_positive_only"
    if fp == 0 and fn > 0:
        return "false_negative_only"
    if fp >= 2 * max(fn, 1):
        return "false_positive_heavy"
    if fn >= 2 * max(fp, 1):
        return "false_negative_heavy"
    return "mixed_fp_fn"


def choose_primary_failure_category(row: dict[str, Any], params: dict[str, Any] | None = None) -> str:
    params = params or {}
    f1 = as_float(row.get("f1"))
    iou = as_float(row.get("iou"))
    precision = as_float(row.get("precision"))
    recall = as_float(row.get("recall"))
    ratio = as_float(row.get("pred_gt_area_ratio"))
    threshold_gap = as_float(row.get("threshold_gap_f1", row.get("threshold_gap")))
    num_pred_blobs = as_int(row.get("pred_num_blobs"))
    num_gt_blobs = as_int(row.get("gt_num_blobs"))

    if f1 >= as_float(params.get("GOOD_F1_THRESHOLD"), 0.85) and iou >= as_float(
        params.get("GOOD_IOU_THRESHOLD"), 0.75
    ):
        return "good_or_minor_error"
    if recall < as_float(params.get("LOW_RECALL_THRESHOLD"), 0.50) and precision >= as_float(
        params.get("LOW_PRECISION_THRESHOLD"), 0.50
    ):
        return "false_negative_missed_tamper"
    if precision < as_float(params.get("LOW_PRECISION_THRESHOLD"), 0.50) and recall >= as_float(
        params.get("LOW_RECALL_THRESHOLD"), 0.50
    ):
        return "false_positive_over_detection"
    if precision < as_float(params.get("LOW_PRECISION_THRESHOLD"), 0.50) and recall < as_float(
        params.get("LOW_RECALL_THRESHOLD"), 0.50
    ):
        return "both_fp_and_fn"
    if ratio > as_float(params.get("OVER_EXPANSION_RATIO"), 1.80):
        return "over_expansion"
    if ratio < as_float(params.get("UNDER_EXPANSION_RATIO"), 0.55):
        return "under_expansion"
    if threshold_gap > as_float(params.get("CALIBRATION_GAP_THRESHOLD"), 0.15):
        return "threshold_calibration_failure"
    if num_pred_blobs > max(
        as_int(params.get("FRAGMENTATION_MULTIPLIER"), 3) * num_gt_blobs,
        as_int(params.get("FRAGMENTATION_MIN_BLOBS"), 5),
    ):
        return "fragmentation_noise"
    return "boundary_or_shape_error"


def connected_component_stats(binary_mask) -> dict[str, float | int]:
    import numpy as np  # type: ignore

    mask = np.asarray(binary_mask).astype(bool)
    if mask.size == 0 or not mask.any():
        return {"num_blobs": 0, "largest_blob_area": 0, "mean_blob_area": 0.0, "p95_blob_area": 0.0}
    try:
        from scipy import ndimage  # type: ignore

        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        labeled, count = ndimage.label(mask, structure=structure)
        areas = np.bincount(labeled.reshape(-1))[1:]
    except Exception:
        areas = _component_areas_bfs(mask)
        count = len(areas)
        areas = np.asarray(areas, dtype=np.int64)
    if count == 0 or areas.size == 0:
        return {"num_blobs": 0, "largest_blob_area": 0, "mean_blob_area": 0.0, "p95_blob_area": 0.0}
    return {
        "num_blobs": int(count),
        "largest_blob_area": int(areas.max()),
        "mean_blob_area": float(areas.mean()),
        "p95_blob_area": float(np.percentile(areas, 95)),
    }


def compute_threshold_sweep_rows(
    prob,
    gt,
    thresholds: Iterable[float],
    prefix: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        threshold_value = round(float(threshold), 6)
        counts = _confusion_counts(prob, gt, threshold_value)
        metrics = metrics_from_counts(counts)
        gt_area = counts["tp"] + counts["fn"]
        pred_area = counts["tp"] + counts["fp"]
        row = dict(prefix or {})
        row.update(
            {
                "threshold": threshold_value,
                **counts,
                **metrics,
                "gt_area": gt_area,
                "pred_area": pred_area,
                "pred_gt_area_ratio": _safe_div(pred_area, gt_area),
            }
        )
        rows.append(row)
    if not rows:
        return rows, {"best_threshold": math.nan, "best_threshold_f1": math.nan, "best_threshold_iou": math.nan}
    best = max(rows, key=lambda row: (as_float(row.get("f1")), as_float(row.get("iou")), -as_float(row.get("threshold"))))
    return rows, {
        "best_threshold": as_float(best.get("threshold")),
        "best_threshold_f1": as_float(best.get("f1")),
        "best_threshold_iou": as_float(best.get("iou")),
    }


def compute_h95_diagnostics(h95, gt, pred, h95_available: bool, eps: float = 1e-8) -> dict[str, Any]:
    if not h95_available or h95 is None:
        return {
            "h95_available": False,
            "h95_mean_inside_gt": math.nan,
            "h95_p95_inside_gt": math.nan,
            "h95_mean_outside_gt": math.nan,
            "h95_p95_outside_gt": math.nan,
            "h95_mean_inside_pred": math.nan,
            "h95_mean_fp": math.nan,
            "h95_mean_fn": math.nan,
            "h95_signal_ratio": math.nan,
            "likely_h95_signal": "not_available_no_h95_model",
        }

    import numpy as np  # type: ignore

    h95_arr = np.asarray(h95, dtype="float32")
    gt_mask = np.asarray(gt).astype(bool)
    pred_mask = np.asarray(pred).astype(bool)
    outside_gt = ~gt_mask
    fp_mask = pred_mask & ~gt_mask
    fn_mask = ~pred_mask & gt_mask
    inside_mean = _region_mean(h95_arr, gt_mask)
    outside_mean = _region_mean(h95_arr, outside_gt)
    ratio = inside_mean / (outside_mean + eps) if _is_finite(inside_mean) and _is_finite(outside_mean) else math.nan
    if not _is_finite(ratio):
        signal = "unknown_h95_signal"
    elif ratio < 1.2:
        signal = "weak_h95_forensic_signal"
    elif ratio < 2.0:
        signal = "moderate_h95_forensic_signal"
    else:
        signal = "strong_h95_forensic_signal"
    return {
        "h95_available": True,
        "h95_mean_inside_gt": inside_mean,
        "h95_p95_inside_gt": _region_percentile(h95_arr, gt_mask, 95),
        "h95_mean_outside_gt": outside_mean,
        "h95_p95_outside_gt": _region_percentile(h95_arr, outside_gt, 95),
        "h95_mean_inside_pred": _region_mean(h95_arr, pred_mask),
        "h95_mean_fp": _region_mean(h95_arr, fp_mask),
        "h95_mean_fn": _region_mean(h95_arr, fn_mask),
        "h95_signal_ratio": ratio,
        "likely_h95_signal": signal,
    }


def categorize_failure(row: dict[str, Any]) -> dict[str, str]:
    tp = as_int(row.get("tp"))
    fp = as_int(row.get("fp"))
    fn = as_int(row.get("fn"))
    gt = tp + fn
    pred = tp + fp
    return {
        "failure_category": choose_raw_pixel_category(tp, fp, fn, gt, pred),
        "failure_severity": severity_from_f1(as_float(row.get("f1"))),
    }


def add_failure_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.update(categorize_failure(row))
    tp = as_int(out.get("tp"))
    fp = as_int(out.get("fp"))
    fn = as_int(out.get("fn"))
    gt_area = tp + fn
    pred_area = tp + fp
    out["raw_pixel_category"] = out["failure_category"]
    out["severity"] = out["failure_severity"]
    out["pred_gt_area_ratio"] = _safe_div(pred_area, gt_area)
    out["threshold_gap_f1"] = as_float(out.get("best_threshold_f1"), as_float(out.get("f1"))) - as_float(out.get("f1"))
    out.setdefault("gt_num_blobs", 0)
    out.setdefault("pred_num_blobs", 0)
    out["primary_failure_category"] = choose_primary_failure_category(out)
    out["precision_recall_pattern"] = _precision_recall_pattern(out)
    out["expansion_type"] = _expansion_type(out)
    out["threshold_gap"] = out["threshold_gap_f1"]
    out["calibration_status"] = (
        "threshold_calibration_failure" if as_float(out["threshold_gap_f1"]) > 0.15 else "threshold_not_main_issue"
    )
    out["fragmentation_status"] = (
        "fragmentation_noise"
        if as_int(out.get("pred_num_blobs")) > max(3 * as_int(out.get("gt_num_blobs")), 5)
        else "not_fragmented"
    )
    out["boundary_shape_status"] = (
        "boundary_or_shape_error"
        if out["primary_failure_category"] == "boundary_or_shape_error"
        else "not_primary_boundary_error"
    )
    out["semantic_failure_status"] = "semantic_failure_possible" if as_float(out.get("f1")) < 0.5 else "not_primary_semantic_failure"
    return out


def run_failure_analysis(config_path: str | Path, checkpoint: str | None = None) -> Path:
    config = load_config(config_path)
    paths = resolve_output_paths(config)
    failure_dir = ensure_dir(paths["failure"])
    run_dir = paths["run"]
    test_sets = dict(deep_get(config, "data.test_sets", {}))
    if not test_sets:
        raise ValueError("No data.test_sets configured for failure analysis")

    params = _failure_params(config)
    thresholds = frange(params["THRESHOLD_MIN"], params["THRESHOLD_MAX"], params["THRESHOLD_STEP"])
    per_split_rows = _load_per_split_metric_rows(run_dir, list(test_sets))
    selected_sources: dict[str, list[dict[str, Any]]] = {}
    for test_set in test_sets:
        ranked = select_worst_k(per_split_rows.get(test_set, []), params["TOP_K_PER_SET"])
        for rank, row in enumerate(ranked, start=1):
            row["test_set"] = test_set
            row["rank_within_test_set"] = rank
        selected_sources[test_set] = ranked

    model = create_model(config)
    runtime = resolve_runtime(config)
    resolved_checkpoint = checkpoint or str(paths["run"] / "checkpoints" / "best_model.pth")
    if Path(resolved_checkpoint).exists() or not bool(deep_get(config, "runtime.dummy", False)):
        _load_checkpoint_if_available(model, resolved_checkpoint, config)

    selected_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    split_outputs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    batch_size = int(deep_get(config, "failure_analysis.batch_size_infer", deep_get(config, "evaluation.batch_size", 4)))
    h95_available = bool(int(deep_get(config, "model.in_channels", 2)) >= 2)

    for test_set, folder in test_sets.items():
        split_dir = ensure_dir(failure_dir / "selected_worst_200" / test_set)
        source_rows = selected_sources[test_set]
        if not source_rows:
            write_csv(split_dir / "worst_200_failure_metrics.csv", [])
            continue

        selected_by_index = {as_int(row.get("dataset_index")): row for row in source_rows}
        indices = [as_int(row.get("dataset_index")) for row in source_rows]
        dataset = make_dataset(config, test_set, folder, indices)
        for batch in iter_batches(dataset, batch_size):
            probs = _predict_failure_batch(model, batch, config, runtime)
            for sample, prob in zip(batch, probs):
                source = selected_by_index[as_int(sample.get("dataset_index"))]
                row, rows_for_thresholds = _analyze_failure_sample(
                    sample=sample,
                    prob=prob,
                    source_row=source,
                    rank=as_int(source.get("rank_within_test_set")),
                    thresholds=thresholds,
                    params=params,
                    h95_available=h95_available,
                    split_dir=split_dir,
                )
                selected_rows.append(row)
                split_outputs[test_set].append(row)
                threshold_rows.extend(rows_for_thresholds)
        write_csv(split_dir / "worst_200_failure_metrics.csv", split_outputs[test_set])

    summaries = _write_failure_outputs(failure_dir, selected_rows, threshold_rows, params, config, resolved_checkpoint)
    _write_plots(failure_dir / "plots", selected_rows)
    _write_failure_report(
        failure_dir / "failure_case_report.md",
        selected_rows,
        threshold_rows,
        summaries,
        params,
        resolved_checkpoint,
    )
    return failure_dir


def _analyze_failure_sample(
    sample: dict[str, Any],
    prob,
    source_row: dict[str, Any],
    rank: int,
    thresholds: list[float],
    params: dict[str, Any],
    h95_available: bool,
    split_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import numpy as np  # type: ignore

    prob_arr = np.asarray(prob, dtype="float32")
    gt = np.asarray(sample["mask"][0]).astype("float32") >= 0.5
    default_threshold = params["DEFAULT_THRESHOLD"]
    pred = prob_arr >= default_threshold
    counts = _confusion_counts(prob_arr, gt, default_threshold)
    metrics = metrics_from_counts(counts)
    total_pixels = sum(counts.values())
    gt_area = counts["tp"] + counts["fn"]
    pred_area = counts["tp"] + counts["fp"]
    error_area = counts["fp"] + counts["fn"]
    threshold_prefix = {
        "test_set": sample.get("test_set"),
        "rank_within_test_set": rank,
        "image_id": sample.get("image_id"),
        "dataset_index": sample.get("dataset_index"),
    }
    threshold_rows, best = compute_threshold_sweep_rows(prob_arr, gt, thresholds, threshold_prefix)
    tp_mask = pred & gt
    fp_mask = pred & ~gt
    fn_mask = ~pred & gt
    row: dict[str, Any] = {
        "test_set": sample.get("test_set"),
        "rank_within_test_set": rank,
        "image_id": sample.get("image_id", source_row.get("image_id", "")),
        "sample_id": source_row.get("sample_id", sample.get("image_id", "")),
        "dataset_index": sample.get("dataset_index"),
        "dataset_folder": sample.get("folder", source_row.get("dataset_folder", "")),
        "postprocess": "no_blob",
        "threshold_at_default": default_threshold,
        "threshold": default_threshold,
        "selection_f1": as_float(source_row.get("f1")),
        "selection_iou": as_float(source_row.get("iou")),
        **counts,
        **metrics,
        "gt_area": gt_area,
        "pred_area": pred_area,
        "pred_gt_area_ratio": _safe_div(pred_area, gt_area),
        "error_area": error_area,
        "error_area_ratio": _safe_div(error_area, total_pixels),
        "gt_area_ratio": _safe_div(gt_area, total_pixels),
        "pred_area_ratio": _safe_div(pred_area, total_pixels),
        **_probability_diagnostics(prob_arr, gt, pred),
        **compute_h95_diagnostics(sample.get("h95"), gt, pred, h95_available, params["H95_RATIO_EPS"]),
        **_prefixed_component_stats("gt", gt),
        **_prefixed_component_stats("pred", pred),
        **_prefixed_component_stats("tp", tp_mask),
        **_prefixed_component_stats("fp", fp_mask),
        **_prefixed_component_stats("fn", fn_mask),
        **best,
    }
    row["threshold_gap_f1"] = as_float(row.get("best_threshold_f1")) - as_float(row.get("f1"))
    row["threshold_gap_iou"] = as_float(row.get("best_threshold_iou")) - as_float(row.get("iou"))
    row["severity"] = severity_from_f1(as_float(row.get("f1")))
    row["failure_severity"] = row["severity"]
    row["raw_pixel_category"] = choose_raw_pixel_category(
        counts["tp"], counts["fp"], counts["fn"], gt_area, pred_area
    )
    row["failure_category"] = row["raw_pixel_category"]
    row["primary_failure_category"] = choose_primary_failure_category(row, params)
    row["precision_recall_pattern"] = _precision_recall_pattern(row)
    row["expansion_type"] = _expansion_type(row)
    row["calibration_status"] = (
        "threshold_calibration_failure"
        if as_float(row.get("threshold_gap_f1")) > params["CALIBRATION_GAP_THRESHOLD"]
        else "threshold_not_main_issue"
    )
    row["fragmentation_status"] = (
        "fragmentation_noise"
        if as_int(row.get("pred_num_blobs")) > max(params["FRAGMENTATION_MULTIPLIER"] * as_int(row.get("gt_num_blobs")), params["FRAGMENTATION_MIN_BLOBS"])
        else "not_fragmented"
    )
    row["boundary_shape_status"] = (
        "boundary_or_shape_error"
        if row["primary_failure_category"] == "boundary_or_shape_error"
        else "not_primary_boundary_error"
    )
    row["semantic_failure_status"] = "semantic_failure_possible" if as_float(row.get("f1")) < 0.5 else "not_primary_semantic_failure"
    row.update(_likely_tamper_fields(sample, row, params))
    row["likely_reasons"] = _likely_reasons(row, params)

    panel_path = split_dir / f"rank_{rank:03d}_f1_{as_float(row.get('f1')):.4f}_{_safe_filename(str(row.get('image_id')))}.png"
    save_failure_diagnostic_panel(panel_path, sample, prob_arr, pred, row, h95_available=h95_available)
    row["panel_path"] = str(panel_path.resolve())
    return row, threshold_rows


def _write_failure_outputs(
    failure_dir: Path,
    selected_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    params: dict[str, Any],
    config: dict[str, Any],
    checkpoint: str,
) -> dict[str, list[dict[str, Any]]]:
    params = dict(params)
    params.update(
        {
            "SELECTED_IMAGES": len(selected_rows),
            "THRESHOLD_SWEEP_ROWS": len(threshold_rows),
            "CHECKPOINT_PATH": checkpoint,
            "SOURCE_CONFIG_PATH": config.get("_config_path", ""),
        }
    )
    write_text(failure_dir / "failure_analysis_config.yaml", dump_config(params))
    write_json(failure_dir / "failure_analysis_config.json", params)
    write_csv(failure_dir / "failure_cases_all_selected.csv", selected_rows)
    write_csv(failure_dir / "threshold_sweep_0.01_0.99.csv", threshold_rows)

    summaries = {
        "test_set": _summary_by_test_set(selected_rows),
        "primary_category": _summary_by(selected_rows, ["test_set", "primary_failure_category", "severity"]),
        "raw_category": _summary_by(selected_rows, ["test_set", "raw_pixel_category", "severity"]),
        "severity": _summary_by(selected_rows, ["test_set", "severity"]),
        "likely_reason": _summary_by_likely_reason(selected_rows),
        "likely_tamper_type": _summary_by(selected_rows, ["test_set", "likely_tamper_type"]),
    }
    write_csv(failure_dir / "failure_summary_by_test_set.csv", summaries["test_set"])
    write_csv(failure_dir / "failure_summary_by_primary_category.csv", summaries["primary_category"])
    write_csv(failure_dir / "failure_summary_by_raw_category.csv", summaries["raw_category"])
    write_csv(failure_dir / "failure_summary_by_severity.csv", summaries["severity"])
    write_csv(failure_dir / "failure_summary_by_likely_reason.csv", summaries["likely_reason"])
    write_csv(failure_dir / "failure_summary_by_category.csv", summaries["primary_category"])
    write_csv(failure_dir / "failure_summary_by_likely_tamper_type.csv", summaries["likely_tamper_type"])
    return summaries


def _load_per_split_metric_rows(run_dir: Path, test_sets: list[str]) -> dict[str, list[dict[str, Any]]]:
    combined_rows: list[dict[str, Any]] | None = None
    out: dict[str, list[dict[str, Any]]] = {}
    per_image_dir = run_dir / "per_image_metrics"
    for test_set in test_sets:
        rows: list[dict[str, Any]] = []
        candidates = [per_image_dir / f"{test_set}_resnet34_h95_no_blob.csv"]
        if per_image_dir.exists():
            candidates.extend(sorted(per_image_dir.glob(f"{test_set}_*_no_blob.csv")))
        for candidate in candidates:
            if candidate.exists():
                rows = read_csv_rows(candidate)
                break
        if not rows:
            if combined_rows is None:
                combined_path = run_dir / "per_image_metrics.csv"
                if not combined_path.exists():
                    raise FileNotFoundError(f"Run evaluation before failure analysis: {combined_path}")
                combined_rows = read_csv_rows(combined_path)
            rows = [
                row
                for row in combined_rows
                if row.get("test_set") == test_set and row.get("postprocess", "no_blob") == "no_blob"
            ]
        out[test_set] = _normalize_per_image_rows(rows, test_set)
    return out


def _normalize_per_image_rows(rows: list[dict[str, Any]], test_set: str) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for row in rows:
        f1 = _coerce_float(row.get("f1"))
        iou = _coerce_float(row.get("iou"))
        dataset_index = _coerce_int(row.get("dataset_index"))
        if f1 is None or iou is None or dataset_index is None:
            continue
        item = dict(row)
        item["test_set"] = test_set
        item["f1"] = f1
        item["iou"] = iou
        item["dataset_index"] = dataset_index
        item.setdefault("image_id", f"{test_set}_{dataset_index:06d}")
        cleaned.append(item)
    return cleaned


def _predict_failure_batch(model, batch: list[dict[str, Any]], config: dict[str, Any], runtime: dict[str, Any]):
    import numpy as np  # type: ignore

    in_channels = int(deep_get(config, "model.in_channels", 2))
    if bool(deep_get(config, "runtime.dummy", False)) or model.__class__.__name__ == "DummyResNetUNet":
        channel = 1 if in_channels >= 2 else 0
        return [np.clip(_model_input_array(sample, in_channels)[channel if channel < in_channels else 0], 0.0, 1.0) for sample in batch]
    try:
        import torch  # type: ignore

        device = torch.device(runtime["device"])
        model.to(device)
        model.eval()
        x = torch.from_numpy(np.stack([_model_input_array(sample, in_channels) for sample in batch])).to(device)
        if runtime["device"] == "cuda" and runtime.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        amp_dtype = torch.bfloat16 if runtime.get("amp_dtype") == "bfloat16" else torch.float16
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=runtime["device"] == "cuda"):
                logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
        return [probs[i] for i in range(probs.shape[0])]
    except Exception as exc:
        raise RuntimeError(f"Failure-analysis inference failed: {exc}") from exc


def _model_input_array(sample: dict[str, Any], in_channels: int):
    import numpy as np  # type: ignore

    arr = np.asarray(sample["input"], dtype="float32")
    if arr.shape[0] == in_channels:
        return arr
    if arr.shape[0] > in_channels:
        return arr[:in_channels]
    padding = np.zeros((in_channels - arr.shape[0], *arr.shape[1:]), dtype="float32")
    return np.concatenate([arr, padding], axis=0)


def _confusion_counts(prob, gt, threshold: float) -> dict[str, int]:
    import numpy as np  # type: ignore

    pred = np.asarray(prob) >= threshold
    gt_mask = np.asarray(gt) >= 0.5
    return {
        "tp": int(np.count_nonzero(pred & gt_mask)),
        "fp": int(np.count_nonzero(pred & ~gt_mask)),
        "fn": int(np.count_nonzero(~pred & gt_mask)),
        "tn": int(np.count_nonzero(~pred & ~gt_mask)),
    }


def _probability_diagnostics(prob, gt, pred) -> dict[str, float]:
    import numpy as np  # type: ignore

    prob_arr = np.asarray(prob, dtype="float32")
    gt_mask = np.asarray(gt).astype(bool)
    pred_mask = np.asarray(pred).astype(bool)
    tp_mask = pred_mask & gt_mask
    fp_mask = pred_mask & ~gt_mask
    fn_mask = ~pred_mask & gt_mask
    return {
        "prob_mean_global": float(prob_arr.mean()) if prob_arr.size else math.nan,
        "prob_p95_global": float(np.percentile(prob_arr, 95)) if prob_arr.size else math.nan,
        "prob_mean_inside_gt": _region_mean(prob_arr, gt_mask),
        "prob_mean_outside_gt": _region_mean(prob_arr, ~gt_mask),
        "prob_mean_inside_pred": _region_mean(prob_arr, pred_mask),
        "prob_mean_tp": _region_mean(prob_arr, tp_mask),
        "prob_mean_fp": _region_mean(prob_arr, fp_mask),
        "prob_mean_fn": _region_mean(prob_arr, fn_mask),
    }


def _prefixed_component_stats(prefix: str, mask) -> dict[str, Any]:
    stats = connected_component_stats(mask)
    return {f"{prefix}_{key}": value for key, value in stats.items()}


def _component_areas_bfs(mask) -> list[int]:
    import numpy as np  # type: ignore

    visited = np.zeros(mask.shape, dtype=bool)
    areas: list[int] = []
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            area = 0
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                area += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            areas.append(area)
    return areas


def _region_mean(arr, mask) -> float:
    import numpy as np  # type: ignore

    values = np.asarray(arr)[np.asarray(mask).astype(bool)]
    return float(values.mean()) if values.size else math.nan


def _region_percentile(arr, mask, percentile: float) -> float:
    import numpy as np  # type: ignore

    values = np.asarray(arr)[np.asarray(mask).astype(bool)]
    return float(np.percentile(values, percentile)) if values.size else math.nan


def _likely_reasons(row: dict[str, Any], params: dict[str, Any]) -> str:
    reasons: list[str] = []
    if as_float(row.get("gt_area")) < params["TINY_TAMPER_AREA"]:
        reasons.append("tiny_tamper_region")
    if as_float(row.get("pred_gt_area_ratio")) > params["PRED_MUCH_LARGER_RATIO"]:
        reasons.append("large_over_prediction")
    if as_float(row.get("pred_gt_area_ratio")) < params["PRED_MUCH_SMALLER_RATIO"]:
        reasons.append("large_under_prediction")
    if as_float(row.get("threshold_gap_f1")) > params["CALIBRATION_GAP_THRESHOLD"]:
        reasons.append("strong_threshold_calibration_issue")
    if as_int(row.get("pred_num_blobs")) > max(
        params["FRAGMENTATION_MULTIPLIER"] * as_int(row.get("gt_num_blobs")),
        params["FRAGMENTATION_MIN_BLOBS"],
    ):
        reasons.append("fragmented_prediction_noise")
    if row.get("h95_available") and _is_finite(row.get("h95_signal_ratio")):
        ratio = as_float(row.get("h95_signal_ratio"))
        if ratio < params["WEAK_H95_RATIO"]:
            reasons.append("weak_h95_forensic_signal")
        if ratio > params["STRONG_H95_RATIO"] and as_float(row.get("f1")) < 0.5:
            reasons.append("strong_h95_but_model_failed")
    if not reasons:
        reasons.append("boundary_alignment_error")
    return "; ".join(reasons)


def _likely_tamper_fields(sample: dict[str, Any], row: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    tamper_type = str(sample.get("tamper_type", row.get("tamper_type", "unknown")) or "unknown")
    tamper_source = str(sample.get("tamper_type_source", row.get("tamper_type_source", "unknown")) or "unknown")
    out = {
        "tamper_type": tamper_type,
        "tamper_type_source": tamper_source,
        "tamper_type_confidence": sample.get("tamper_type_confidence", row.get("tamper_type_confidence", "")),
        "tamper_type_reason": sample.get("tamper_type_reason", row.get("tamper_type_reason", "")),
        "metadata_tamper_type": tamper_type if tamper_source == "metadata" and tamper_type != "unknown" else "",
        "heuristic_tiny_tamper_region": as_float(row.get("gt_area")) < params["TINY_TAMPER_AREA"],
        "heuristic_strong_h95_signal_inside_gt": bool(
            row.get("h95_available") and _is_finite(row.get("h95_signal_ratio")) and as_float(row.get("h95_signal_ratio")) > params["STRONG_H95_RATIO"]
        ),
        "heuristic_prediction_much_larger_than_gt": as_float(row.get("pred_gt_area_ratio")) > params["PRED_MUCH_LARGER_RATIO"],
        "heuristic_prediction_much_smaller_than_gt": as_float(row.get("pred_gt_area_ratio")) < params["PRED_MUCH_SMALLER_RATIO"],
    }
    if out["metadata_tamper_type"]:
        out["likely_tamper_type"] = out["metadata_tamper_type"]
        out["likely_tamper_confidence"] = "metadata"
    elif out["heuristic_tiny_tamper_region"]:
        out["likely_tamper_type"] = "tiny_tamper_region"
        out["likely_tamper_confidence"] = "heuristic"
    elif out["heuristic_strong_h95_signal_inside_gt"]:
        out["likely_tamper_type"] = "strong_h95_signal_inside_gt"
        out["likely_tamper_confidence"] = "heuristic"
    elif out["heuristic_prediction_much_larger_than_gt"]:
        out["likely_tamper_type"] = "prediction_much_larger_than_gt"
        out["likely_tamper_confidence"] = "heuristic"
    elif out["heuristic_prediction_much_smaller_than_gt"]:
        out["likely_tamper_type"] = "prediction_much_smaller_than_gt"
        out["likely_tamper_confidence"] = "heuristic"
    else:
        out["likely_tamper_type"] = "heuristic_uncertain"
        out["likely_tamper_confidence"] = "low"
    return out


def _summary_by_test_set(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("test_set", "unknown"))].append(row)
    out: list[dict[str, Any]] = []
    for test_set, group in grouped.items():
        entry = _aggregate_group(group)
        entry["test_set"] = test_set
        entry["selected_images"] = len(group)
        entry["min_f1"] = _min_value(group, "f1")
        entry["max_f1"] = _max_value(group, "f1")
        for severity in SEVERITY_LEVELS:
            entry[f"{severity}_count"] = sum(1 for row in group if row.get("severity") == severity)
        out.append(entry)
    return sorted(out, key=lambda row: str(row.get("test_set")))


def _summary_by(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "unknown") for key in keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_values, group in grouped.items():
        entry = {key: value for key, value in zip(keys, key_values)}
        entry.update(_aggregate_group(group))
        entry["selected_images"] = len(group)
        out.append(entry)
    return sorted(out, key=lambda row: tuple(str(row.get(key, "")) for key in keys))


def _summary_by_likely_reason(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    exploded: list[dict[str, Any]] = []
    for row in rows:
        for reason in str(row.get("likely_reasons", "unknown")).split(";"):
            item = dict(row)
            item["likely_reason"] = reason.strip() or "unknown"
            exploded.append(item)
    return _summary_by(exploded, ["test_set", "likely_reason"])


def _aggregate_group(group: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: sum(as_int(row.get(key)) for row in group) for key in ["tp", "fp", "fn", "tn"]}
    return {
        **counts,
        **metrics_from_counts(counts),
        "mean_f1": _mean_value(group, "f1"),
        "mean_iou": _mean_value(group, "iou"),
        "mean_precision": _mean_value(group, "precision"),
        "mean_recall": _mean_value(group, "recall"),
        "mean_error_area_ratio": _mean_value(group, "error_area_ratio"),
        "mean_pred_gt_area_ratio": _mean_value(group, "pred_gt_area_ratio"),
        "mean_h95_signal_ratio": _mean_value(group, "h95_signal_ratio"),
    }


def _write_plots(plot_dir: Path, rows: list[dict[str, Any]]) -> None:
    ensure_dir(plot_dir)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        _plot_hist(plt, plot_dir / "worst200_f1_distribution.png", rows, "f1", "Worst-case F1 distribution", "F1")
        _plot_counter(
            plt,
            plot_dir / "failure_category_counts.png",
            Counter(str(row.get("primary_failure_category", "unknown")) for row in rows),
            "Primary failure category counts",
        )
        _plot_grouped_counter(
            plt,
            plot_dir / "severity_counts_by_test_set.png",
            rows,
            "test_set",
            "severity",
            "Severity counts by test set",
        )
        _plot_counter(
            plt,
            plot_dir / "raw_category_counts.png",
            Counter(str(row.get("raw_pixel_category", "unknown")) for row in rows),
            "Raw pixel-error category counts",
        )
        _plot_scatter(
            plt,
            plot_dir / "precision_recall_scatter.png",
            rows,
            "recall",
            "precision",
            "Recall",
            "Precision",
            "Precision vs recall for selected failures",
        )
        _plot_hist(
            plt,
            plot_dir / "threshold_gap_distribution.png",
            rows,
            "threshold_gap_f1",
            "Best-threshold F1 gain distribution",
            "Best F1 - F1 at 0.5",
        )
        _plot_scatter(
            plt,
            plot_dir / "pred_gt_area_ratio_vs_f1.png",
            rows,
            "pred_gt_area_ratio",
            "f1",
            "Predicted / GT area ratio",
            "F1",
            "Area ratio vs F1",
        )
        _plot_scatter(
            plt,
            plot_dir / "h95_signal_ratio_vs_f1.png",
            rows,
            "h95_signal_ratio",
            "f1",
            "H95 inside/outside GT ratio",
            "F1",
            "H95 signal ratio vs F1",
        )
        _plot_confusion_totals(plt, plot_dir / "confusion_totals_by_test_set.png", rows)
    except Exception:
        for name in FAILURE_PLOT_NAMES:
            _write_placeholder_png(plot_dir / name, name)


def _write_failure_report(
    path: Path,
    selected: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    summaries: dict[str, list[dict[str, Any]]],
    params: dict[str, Any],
    checkpoint: str,
) -> None:
    lines = [
        "# Failure Case Analysis Report",
        "",
        "## 1. Scope",
        "- model name: ResNet34-H95",
        f"- checkpoint path: {checkpoint}",
        f"- test sets: {', '.join(params['TEST_SETS'])}",
        f"- selected images: {len(selected)}",
        "- selection rule: f1 ascending, iou ascending, dataset_index ascending",
        f"- threshold sweep range: {params['THRESHOLD_MIN']:.2f}-{params['THRESHOLD_MAX']:.2f} step {params['THRESHOLD_STEP']:.2f}",
        "",
        "## 2. Summary By Test Set",
        _markdown_table(
            summaries.get("test_set", []),
            ["test_set", "selected_images", "mean_f1", "mean_iou", "mean_precision", "mean_recall", "mean_h95_signal_ratio"],
        ),
        "",
        "## 3. Severity Breakdown",
        _markdown_table(summaries.get("severity", []), ["test_set", "severity", "selected_images", "mean_f1", "mean_iou"]),
        "",
        "## 4. Primary Failure Categories",
        _markdown_table(
            summaries.get("primary_category", []),
            ["test_set", "primary_failure_category", "severity", "selected_images", "mean_f1", "mean_precision", "mean_recall"],
            limit=20,
        ),
        "",
        "## 5. Raw Pixel-Error Categories",
        _markdown_table(
            summaries.get("raw_category", []),
            ["test_set", "raw_pixel_category", "severity", "selected_images", "mean_f1"],
            limit=20,
        ),
        "",
        "## 6. H95 Diagnostic Findings",
        _h95_report_lines(selected),
        "",
        "## 7. Calibration Findings",
        _calibration_report_lines(selected, threshold_rows),
        "",
        "## 8. Most Important Failure Modes",
        _markdown_table(summaries.get("likely_reason", []), ["test_set", "likely_reason", "selected_images", "mean_f1"], limit=20),
        "",
        "## 9. Recommended Improvements",
        "- calibrate threshold per deployment domain when threshold gaps are consistently positive",
        "- prioritize false-positive suppression for over-detection categories",
        "- improve tiny-region sensitivity where tiny tamper regions dominate likely reasons",
        "- inspect SCD and TestingSet domain-specific failures separately before changing training data",
        "",
        "## 10. Artifact List",
        "- failure_cases_all_selected.csv",
        "- threshold_sweep_0.01_0.99.csv",
        "- failure_summary_by_test_set.csv",
        "- failure_summary_by_primary_category.csv",
        "- failure_summary_by_raw_category.csv",
        "- failure_summary_by_severity.csv",
        "- failure_summary_by_likely_reason.csv",
        "- selected_worst_200/<test_set>/rank_*.png",
        "- plots/*.png",
        "",
        "## 11. Visualization Artifacts",
    ]
    lines.extend(f"- plots/{name}" for name in FAILURE_PLOT_NAMES)
    write_text(path, "\n".join(lines) + "\n")


def _failure_params(config: dict[str, Any]) -> dict[str, Any]:
    test_sets = list(dict(deep_get(config, "data.test_sets", {})).keys())
    return {
        "TEST_SETS": test_sets,
        "TOP_K_PER_SET": int(deep_get(config, "failure_analysis.top_k", 200)),
        "PRIMARY_SCENARIO": "no_blob",
        "SORT_KEYS": ["f1", "iou", "dataset_index"],
        "DEFAULT_THRESHOLD": float(deep_get(config, "failure_analysis.default_threshold", deep_get(config, "evaluation.official_threshold", 0.5))),
        "THRESHOLD_MIN": float(deep_get(config, "failure_analysis.threshold_min", deep_get(config, "evaluation.threshold_min", 0.01))),
        "THRESHOLD_MAX": float(deep_get(config, "failure_analysis.threshold_max", deep_get(config, "evaluation.threshold_max", 0.99))),
        "THRESHOLD_STEP": float(deep_get(config, "failure_analysis.threshold_step", deep_get(config, "evaluation.threshold_step", 0.01))),
        "EPS": 1e-8,
        "GOOD_F1_THRESHOLD": 0.85,
        "GOOD_IOU_THRESHOLD": 0.75,
        "LOW_RECALL_THRESHOLD": 0.50,
        "LOW_PRECISION_THRESHOLD": 0.50,
        "OVER_EXPANSION_RATIO": 1.80,
        "UNDER_EXPANSION_RATIO": 0.55,
        "CALIBRATION_GAP_THRESHOLD": 0.15,
        "FRAGMENTATION_MULTIPLIER": 3,
        "FRAGMENTATION_MIN_BLOBS": 5,
        "TINY_TAMPER_AREA": 500,
        "PRED_MUCH_LARGER_RATIO": 3.0,
        "PRED_MUCH_SMALLER_RATIO": 0.30,
        "STRONG_H95_RATIO": 2.0,
        "WEAK_H95_RATIO": 1.2,
        "H95_RATIO_EPS": 1e-8,
        "CONNECTIVITY": 4,
        "PANEL_LAYOUT": "2x5",
        "SAVE_PANELS": True,
        "SAVE_THRESHOLD_SWEEP": True,
        "PLOTS": FAILURE_PLOT_NAMES,
    }


def _precision_recall_pattern(row: dict[str, Any]) -> str:
    precision = as_float(row.get("precision"))
    recall = as_float(row.get("recall"))
    if recall < 0.5 and precision >= 0.5:
        return "false_negative_missed_tamper"
    if precision < 0.5 and recall >= 0.5:
        return "false_positive_over_detection"
    if precision < 0.5 and recall < 0.5:
        return "both_fp_and_fn"
    return "balanced_or_minor"


def _expansion_type(row: dict[str, Any]) -> str:
    ratio = as_float(row.get("pred_gt_area_ratio"))
    if ratio > 1.8:
        return "over_expansion"
    if ratio < 0.55:
        return "under_expansion"
    return "area_reasonable"


def _plot_hist(plt, path: Path, rows: list[dict[str, Any]], key: str, title: str, xlabel: str) -> None:
    values = _finite_column(rows, key)
    plt.figure(figsize=(8, 4.8))
    plt.hist(values or [0.0], bins=min(30, max(5, len(values) // 3 if values else 5)), color="#4477AA", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Selected images")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_counter(plt, path: Path, counter: Counter[str], title: str) -> None:
    labels, values = zip(*counter.most_common()) if counter else (["none"], [0])
    plt.figure(figsize=(9, 5))
    plt.bar(range(len(labels)), values, color="#66A61E")
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.title(title)
    plt.ylabel("Selected images")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_grouped_counter(plt, path: Path, rows: list[dict[str, Any]], group_key: str, value_key: str, title: str) -> None:
    groups = sorted({str(row.get(group_key, "unknown")) for row in rows}) or ["none"]
    values = sorted({str(row.get(value_key, "unknown")) for row in rows}) or ["none"]
    counts = Counter((str(row.get(group_key, "unknown")), str(row.get(value_key, "unknown"))) for row in rows)
    bottom = [0] * len(groups)
    plt.figure(figsize=(8, 5))
    for value in values:
        series = [counts[(group, value)] for group in groups]
        plt.bar(groups, series, bottom=bottom, label=value)
        bottom = [left + right for left, right in zip(bottom, series)]
    plt.title(title)
    plt.ylabel("Selected images")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_scatter(plt, path: Path, rows: list[dict[str, Any]], x_key: str, y_key: str, xlabel: str, ylabel: str, title: str) -> None:
    points = [(as_float(row.get(x_key), math.nan), as_float(row.get(y_key), math.nan)) for row in rows]
    points = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    x_values = [point[0] for point in points] or [0.0]
    y_values = [point[1] for point in points] or [0.0]
    plt.figure(figsize=(7, 5))
    plt.scatter(x_values, y_values, s=18, alpha=0.7, color="#D55E00")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_confusion_totals(plt, path: Path, rows: list[dict[str, Any]]) -> None:
    groups = sorted({str(row.get("test_set", "unknown")) for row in rows}) or ["none"]
    keys = ["tp", "fp", "fn"]
    colors = {"tp": "#009E73", "fp": "#D55E00", "fn": "#0072B2"}
    bottom = [0] * len(groups)
    plt.figure(figsize=(8, 5))
    for key in keys:
        series = [sum(as_int(row.get(key)) for row in rows if str(row.get("test_set", "unknown")) == group) for group in groups]
        plt.bar(groups, series, bottom=bottom, label=key.upper(), color=colors[key])
        bottom = [left + right for left, right in zip(bottom, series)]
    plt.title("Confusion totals by test set")
    plt.ylabel("Pixels")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_placeholder_png(path: Path, label: str) -> None:
    try:
        from PIL import Image, ImageDraw  # type: ignore

        image = Image.new("RGB", (800, 450), "white")
        draw = ImageDraw.Draw(image)
        draw.text((24, 24), label, fill="black")
        image.save(path)
    except Exception:
        write_text(path, label)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str], limit: int | None = None) -> str:
    selected = rows[:limit] if limit else rows
    if not selected:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(_format_md_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def _h95_report_lines(rows: list[dict[str, Any]]) -> str:
    available = [row for row in rows if str(row.get("h95_available")).lower() == "true"]
    if not available:
        return "- H95 diagnostics are unavailable for this model/configuration."
    weak = sum(1 for row in available if _is_finite(row.get("h95_signal_ratio")) and as_float(row.get("h95_signal_ratio")) < 1.2)
    strong = sum(1 for row in available if _is_finite(row.get("h95_signal_ratio")) and as_float(row.get("h95_signal_ratio")) > 2.0)
    return "\n".join(
        [
            f"- H95 available rows: {len(available)}",
            f"- weak H95 forensic signal rows: {weak}",
            f"- strong H95 forensic signal rows: {strong}",
            f"- mean H95 signal ratio: {_mean_value(available, 'h95_signal_ratio'):.4f}",
        ]
    )


def _calibration_report_lines(rows: list[dict[str, Any]], threshold_rows: list[dict[str, Any]]) -> str:
    strong = [row for row in rows if as_float(row.get("threshold_gap_f1")) > 0.15]
    return "\n".join(
        [
            f"- threshold sweep rows: {len(threshold_rows)}",
            f"- strong calibration failures: {len(strong)}",
            f"- mean threshold F1 gap: {_mean_value(rows, 'threshold_gap_f1'):.4f}",
        ]
    )


def _format_md_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _mean_value(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_column(rows, key)
    return statistics.fmean(values) if values else math.nan


def _min_value(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_column(rows, key)
    return min(values) if values else math.nan


def _max_value(rows: list[dict[str, Any]], key: str) -> float:
    values = _finite_column(rows, key)
    return max(values) if values else math.nan


def _finite_column(rows: list[dict[str, Any]], key: str) -> list[float]:
    values = [as_float(row.get(key), math.nan) for row in rows]
    return [value for value in values if math.isfinite(value)]


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "sample"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    run_failure_analysis(args.config, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
