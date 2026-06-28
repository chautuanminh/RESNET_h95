from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import deep_get, dump_config, load_config, resolve_output_paths
from .datasets import discover_indices, iter_batches, make_dataset
from .evaluate import _load_checkpoint_if_available, _predict_batch
from .failure_analysis import add_failure_diagnostics, select_worst_k
from .gpu import resolve_runtime
from .metrics import StreamingBinaryMetrics, ThresholdSweep, metrics_from_counts, per_image_metric_row
from .models import create_model
from .tamper_types import assign_tamper_type, load_tamper_metadata
from .utils import as_float, ensure_dir, frange, write_csv, write_text
from .visualization import save_diagnostic_panel, save_placeholder_plot


STANDARD_TYPE_DIRS = ["copy_move", "splicing", "generation", "heuristic_uncertain", "unknown"]


def run_tampering_type_analysis(config_path: str | Path, checkpoint: str | None = None) -> Path:
    config = load_config(config_path)
    paths = resolve_output_paths(config)
    out_dir = ensure_dir(paths["tamper"])
    write_text(out_dir / "tampering_type_analysis_config.yaml", dump_config(config))
    for split in ["TestingSet", "FCD", "SCD"]:
        for type_dir in STANDARD_TYPE_DIRS:
            ensure_dir(out_dir / "examples" / split / type_dir)
            ensure_dir(out_dir / "failures" / split / type_dir)

    model = create_model(config)
    runtime = resolve_runtime(config)
    _load_checkpoint_if_available(model, checkpoint, config)
    metadata = load_tamper_metadata(deep_get(config, "data.tamper_metadata_dir", "tampering_types"))

    thresholds = frange(
        float(deep_get(config, "tampering_type_analysis.threshold_min", 0.01)),
        float(deep_get(config, "tampering_type_analysis.threshold_max", 0.99)),
        float(deep_get(config, "tampering_type_analysis.threshold_step", 0.01)),
    )
    official_threshold = float(deep_get(config, "tampering_type_analysis.official_threshold", 0.5))
    batch_size = int(deep_get(config, "evaluation.batch_size", deep_get(config, "training.batch_size", 4)))
    save_examples = int(deep_get(config, "tampering_type_analysis.save_examples_per_type", 24))
    worst_k = int(deep_get(config, "tampering_type_analysis.worst_k_per_type", 50))
    enable_heuristic = bool(deep_get(config, "tampering_type_analysis.enable_heuristic_classifier", True))

    per_image: list[dict[str, Any]] = []
    stream_by_key: dict[tuple[str, str], StreamingBinaryMetrics] = {}
    sweep_by_key: dict[tuple[str, str], ThresholdSweep] = {}
    examples_seen: dict[tuple[str, str], int] = defaultdict(int)
    samples_for_failure: dict[tuple[str, int], tuple[dict[str, Any], Any, Any, dict[str, Any]]] = {}

    for label, folder in dict(deep_get(config, "data.test_sets", {})).items():
        dataset = make_dataset(config, label, folder, discover_indices(config, folder))
        for batch in iter_batches(dataset, batch_size):
            probs = _predict_batch(model, batch, config, runtime)
            for sample, prob in zip(batch, probs):
                features = _features(sample, prob, official_threshold)
                assignment = assign_tamper_type(
                    label,
                    int(sample["dataset_index"]),
                    metadata if bool(deep_get(config, "tampering_type_analysis.use_metadata_if_available", True)) else {},
                    row=sample,
                    enable_heuristic=enable_heuristic,
                    heuristic_features=features,
                )
                sample = dict(sample)
                sample.update(
                    {
                        "tamper_type": assignment.tamper_type,
                        "tamper_type_source": assignment.tamper_type_source,
                        "tamper_type_confidence": assignment.tamper_type_confidence,
                        "tamper_type_reason": assignment.tamper_type_reason,
                    }
                )
                key = (label, assignment.tamper_type)
                stream_by_key.setdefault(key, StreamingBinaryMetrics(official_threshold)).update(prob, sample["mask"][0])
                sweep_by_key.setdefault(key, ThresholdSweep(thresholds)).update(prob, sample["mask"][0])
                row = per_image_metric_row(prob, sample["mask"][0], official_threshold)
                row.update(
                    {
                        "test_set": label,
                        "dataset_folder": folder,
                        "dataset_index": sample["dataset_index"],
                        "image_id": sample["image_id"],
                        "tamper_type": assignment.tamper_type,
                        "tamper_type_source": assignment.tamper_type_source,
                        "tamper_type_confidence": assignment.tamper_type_confidence,
                        "tamper_type_reason": assignment.tamper_type_reason,
                        **features,
                    }
                )
                per_image.append(row)
                pred = (prob >= official_threshold).astype("uint8")
                samples_for_failure[(label, int(sample["dataset_index"]))] = (sample, prob, pred, row)
                type_dir = _type_dir(assignment.tamper_type)
                if examples_seen[key] < save_examples:
                    save_diagnostic_panel(
                        out_dir
                        / "examples"
                        / label
                        / type_dir
                        / f"sample_{int(sample['dataset_index']):06d}_{assignment.tamper_type}_f1_{float(row['f1']):.4f}.png",
                        sample,
                        prob,
                        pred,
                        row,
                        title="Tampering type example",
                    )
                    examples_seen[key] += 1

    summary_by_test_set = _summary(per_image, ["test_set"])
    summary_by_type = _summary(per_image, ["tamper_type"])
    summary_by_both = _summary(per_image, ["test_set", "tamper_type"])
    threshold_rows = []
    best_rows = []
    for (test_set, tamper_type), sweep in sweep_by_key.items():
        rows = sweep.rows({"test_set": test_set, "tamper_type": tamper_type, "analysis_only": True})
        threshold_rows.extend(rows)
        best_rows.append({**sweep.best_by_f1(), "test_set": test_set, "tamper_type": tamper_type, "analysis_only": True})

    failure_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_image:
        grouped[(row["test_set"], row["tamper_type"])].append(row)
    for (test_set, tamper_type), rows in grouped.items():
        for rank, row in enumerate(select_worst_k(rows, worst_k), start=1):
            failure = add_failure_diagnostics(row)
            failure["rank_within_test_set_and_type"] = rank
            failure_rows.append(failure)
            key = (test_set, int(row["dataset_index"]))
            if key in samples_for_failure:
                sample, prob, pred, metrics = samples_for_failure[key]
                save_diagnostic_panel(
                    out_dir
                    / "failures"
                    / test_set
                    / _type_dir(tamper_type)
                    / f"rank_{rank:03d}_idx_{int(row['dataset_index']):06d}_f1_{float(row['f1']):.4f}.png",
                    sample,
                    prob,
                    pred,
                    metrics,
                    title="Tampering type failure",
                )

    write_csv(out_dir / "tampering_type_per_image.csv", per_image)
    write_csv(out_dir / "tampering_type_summary_by_test_set.csv", summary_by_test_set)
    write_csv(out_dir / "tampering_type_summary_by_type.csv", summary_by_type)
    write_csv(out_dir / "tampering_type_summary_by_test_set_and_type.csv", summary_by_both)
    write_csv(out_dir / "tampering_type_threshold_sweep.csv", threshold_rows)
    write_csv(out_dir / "tampering_type_best_thresholds.csv", best_rows)
    write_csv(out_dir / "tampering_type_failure_summary.csv", _summary(failure_rows, ["test_set", "tamper_type", "failure_category"]))
    write_csv(out_dir / "tampering_type_feature_summary.csv", _feature_summary(per_image))
    _write_plots(out_dir / "plots")
    _write_report(out_dir / "tampering_type_analysis_report.md", per_image, summary_by_both, best_rows)
    _write_summary(paths["root"] / "TAMPERING_TYPE_ANALYSIS_SUMMARY.md", per_image)
    _write_summary(out_dir / "TAMPERING_TYPE_ANALYSIS_SUMMARY.md", per_image)
    return out_dir


def _features(sample: dict[str, Any], prob, threshold: float) -> dict[str, float]:
    import numpy as np  # type: ignore

    mask = np.asarray(sample["mask"][0]) >= 0.5
    pred = np.asarray(prob) >= threshold
    h95 = np.asarray(sample["h95"], dtype="float32")
    total = float(mask.size)
    h95_inside = float(h95[mask].mean()) if mask.any() else 0.0
    h95_outside = float(h95[~mask].mean()) if (~mask).any() else 0.0
    return {
        "mask_area_ratio": float(mask.sum() / total),
        "predicted_area_ratio": float(pred.sum() / total),
        "error_area_ratio": float((pred != mask).sum() / total),
        "component_count": 1.0 if mask.any() else 0.0,
        "patch_similarity": 0.0,
        "h95_mean_inside_gt": h95_inside,
        "h95_mean_outside_gt": h95_outside,
        "h95_inside_outside_ratio": h95_inside / (h95_outside + 1e-8),
    }


def _summary(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key, "unknown") for key in keys)].append(row)
    out = []
    for key_values, group_rows in grouped.items():
        counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for row in group_rows:
            for count_key in counts:
                counts[count_key] += int(float(row.get(count_key, 0)))
        f1s = [as_float(row.get("f1")) for row in group_rows]
        ious = [as_float(row.get("iou")) for row in group_rows]
        precisions = [as_float(row.get("precision")) for row in group_rows]
        recalls = [as_float(row.get("recall")) for row in group_rows]
        entry = {key: value for key, value in zip(keys, key_values)}
        entry.update(
            {
                "num_images": len(group_rows),
                "total_positive_gt_pixels": counts["tp"] + counts["fn"],
                "total_predicted_positive_pixels": counts["tp"] + counts["fp"],
                **counts,
                **metrics_from_counts(counts),
                "mean_f1": statistics.fmean(f1s) if f1s else 0.0,
                "median_f1": statistics.median(f1s) if f1s else 0.0,
                "std_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
                "mean_iou": statistics.fmean(ious) if ious else 0.0,
                "median_iou": statistics.median(ious) if ious else 0.0,
                "std_iou": statistics.pstdev(ious) if len(ious) > 1 else 0.0,
                "mean_precision": statistics.fmean(precisions) if precisions else 0.0,
                "mean_recall": statistics.fmean(recalls) if recalls else 0.0,
                "mean_mask_area_ratio": statistics.fmean(as_float(row.get("mask_area_ratio")) for row in group_rows),
                "mean_predicted_area_ratio": statistics.fmean(as_float(row.get("predicted_area_ratio")) for row in group_rows),
                "mean_error_area_ratio": statistics.fmean(as_float(row.get("error_area_ratio")) for row in group_rows),
            }
        )
        out.append(entry)
    return out


def _feature_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("tamper_type", "unknown"))].append(row)
    out = []
    for tamper_type, group_rows in grouped.items():
        out.append(
            {
                "tamper_type": tamper_type,
                "num_images": len(group_rows),
                "mean_mask_area_ratio": statistics.fmean(as_float(row.get("mask_area_ratio")) for row in group_rows),
                "mean_h95_inside_outside_ratio": statistics.fmean(
                    as_float(row.get("h95_inside_outside_ratio")) for row in group_rows
                ),
                "mean_error_area_ratio": statistics.fmean(as_float(row.get("error_area_ratio")) for row in group_rows),
            }
        )
    return out


def _write_plots(plot_dir: Path) -> None:
    for name in [
        "f1_by_tamper_type.png",
        "iou_by_tamper_type.png",
        "precision_by_tamper_type.png",
        "recall_by_tamper_type.png",
        "count_by_tamper_type.png",
        "error_area_by_tamper_type.png",
        "threshold_sweep_by_tamper_type.png",
    ]:
        save_placeholder_plot(plot_dir / name, name)


def _write_report(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]], best_rows: list[dict[str, Any]]) -> None:
    sources = sorted({str(row.get("tamper_type_source", "unknown")) for row in rows})
    lines = [
        "# Tampering Type Analysis Report",
        "",
        "## Purpose",
        "This diagnostic analysis groups ResNet34-H95 segmentation performance by known or likely tampering type.",
        "",
        "## Metadata Warning",
        "Tamper-type analysis depends on metadata availability. If metadata is missing, heuristic labels are not ground truth.",
        "",
        f"- tamper type sources found: {sources}",
        "",
        "## Counts And Official Threshold Performance",
        "| test_set | tamper_type | images | f1 | iou | precision | recall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row.get('test_set', '')} | {row.get('tamper_type', '')} | {row.get('num_images', 0)} | {float(row.get('f1', 0.0)):.4f} | {float(row.get('iou', 0.0)):.4f} | {float(row.get('precision', 0.0)):.4f} | {float(row.get('recall', 0.0)):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic Thresholds",
            "Best thresholds are analysis-only and do not replace the official 0.5 result.",
            f"- best-threshold rows: {len(best_rows)}",
            "",
            "## Failure Patterns",
            "Failure panels are grouped by test set and tampering type. Each panel includes image index, tamper type, source, confidence, F1, IoU, precision, recall, GT area ratio, predicted area ratio, and error category where available.",
            "",
            "## H95 Interpretation",
            "H95 can help when residual signal aligns with the tampered region; it can struggle when compression residuals are weak inside the ground truth or strong outside it.",
            "",
            "## Limitations",
            "- if metadata is missing, heuristic labels are not ground truth",
            "- no OCR is used",
            "- copy-move, splicing, and generation may be visually ambiguous",
            "- conclusions from heuristic labels must be treated as tentative",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    grouped = _summary(rows, ["tamper_type"]) if rows else []
    easiest = max(grouped, key=lambda row: row.get("f1", 0.0)) if grouped else {}
    hardest = min(grouped, key=lambda row: row.get("f1", 0.0)) if grouped else {}
    lines = [
        "# Tampering Type Analysis Summary",
        "",
        "- Tamper types are obtained from official metadata pickles when available, then existing metric columns, then non-OCR heuristics, then unknown.",
        f"- easiest type by official F1: {easiest.get('tamper_type', 'n/a')}",
        f"- hardest type by official F1: {hardest.get('tamper_type', 'n/a')}",
        "- Failure patterns are grouped in the tampering type analysis folder.",
        "- These diagnostics indicate where the grayscale + H95 residual method aligns or struggles by manipulation type.",
        "",
    ]
    write_text(path, "\n".join(lines))


def _type_dir(tamper_type: str) -> str:
    if tamper_type in {"heuristic_copy_move", "heuristic_splicing", "heuristic_generation"}:
        return tamper_type
    return tamper_type if tamper_type in STANDARD_TYPE_DIRS else "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    run_tampering_type_analysis(args.config, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
