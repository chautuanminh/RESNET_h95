from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import ensure_dir, write_minimal_pdf, write_text


def write_protocol_documents(root: str | Path) -> None:
    root_path = ensure_dir(root)
    write_text(
        root_path / "preprocessing_protocol.md",
        "\n".join(
            [
                "# ResNet34-H95 Preprocessing Protocol",
                "",
                "- Load the original image and convert it to grayscale uint8.",
                "- Resize the grayscale uint8 image to 512x512 with bilinear interpolation for the full experiment.",
                "- Compute the JPEG Q95 recompression residual from the resized uint8 grayscale image.",
                "- Residual is `abs(resized_gray - jpeg_q95_resized_gray)` normalized by `percentile_99 + 1e-8` and clipped to `[0, 1]`.",
                "- Resize masks to 512x512 with nearest-neighbor interpolation only.",
                "- Final model input is `[2, 512, 512]`: channel 0 grayscale, channel 1 H95.",
                "- Final mask is `[1, 512, 512]`.",
                "",
                "Method name: grayscale + H95.",
                "",
            ]
        ),
    )
    write_text(
        root_path / "metric_computation_protocol.md",
        "\n".join(
            [
                "# Metric Computation Protocol",
                "",
                "- Official threshold metrics at 0.5 use exact streaming TP/FP/FN/TN accumulation.",
                "- Threshold sweep metrics from 0.01 to 0.99 use exact streaming TP/FP/FN/TN accumulation per threshold.",
                "- Per-image metrics are saved as compact scalar rows.",
                "- Full test-set probability arrays are never concatenated or retained in memory.",
                "- AUROC/AUPRC are computed with a bounded histogram approximation when full-pixel exact curves would exceed memory limits.",
                "",
            ]
        ),
    )
    write_text(
        root_path / "threshold_policy.md",
        "\n".join(
            [
                "# Threshold Policy",
                "",
                "- Official default threshold: 0.5.",
                "- Diagnostic threshold sweep: 0.01 to 0.99.",
                "- Best-threshold metrics are analysis-only and must not replace official threshold 0.5 metrics.",
                "- TestingSet, FCD, and SCD must not be used for checkpoint selection, early stopping, training decisions, or hyperparameter decisions.",
                "",
            ]
        ),
    )


def write_root_plan(root: str | Path) -> None:
    write_text(
        Path(root) / "PLAN.md",
        "\n".join(
            [
                "# ResNet34-H95 DocTamper Plan",
                "",
                "Train and evaluate a ResNet34-UNet binary segmentation model using a two-channel grayscale + H95 input.",
                "The recorded method name is grayscale + H95.",
                "The experiment enforces a deterministic internal validation split, final-only official evaluation sets, raw thresholded predictions, streaming metrics, failure analysis, and tampering type diagnostics.",
                "",
            ]
        ),
    )


def write_output_contract(root: str | Path) -> None:
    write_text(
        Path(root) / "output.md",
        "\n".join(
            [
                "# Output Contract",
                "",
                "- Root outputs include protocol docs, reports, validation report, model summary, and summary markdown.",
                "- Main run outputs include config snapshot, split CSVs, train/validation metrics, training curves, official metrics, threshold sweep, per-image metrics, checkpoints, plots, and examples.",
                "- Failure analysis outputs include selected worst cases, summaries, plots, and diagnostic panels.",
                "- Tampering type analysis outputs include grouped metrics, threshold diagnostics, plots, examples, failures, and a summary markdown.",
                "- No Jupyter notebooks are generated.",
                "- No connected-component post-processing is applied to predictions or metrics.",
                "",
            ]
        ),
    )


def write_pdf_report(root: str | Path, summary: str = "ResNet34-H95 evaluation report") -> None:
    write_minimal_pdf(Path(root) / "resnet34_h95_evaluation_report.pdf", "ResNet34-H95 Evaluation Report", summary)


def write_res_summary(root: str | Path, summary_rows: list[dict[str, Any]] | None = None) -> None:
    lines = ["# Result Summary", ""]
    if summary_rows:
        lines.append("| test_set | f1 | iou | precision | recall |")
        lines.append("|---|---:|---:|---:|---:|")
        for row in summary_rows:
            lines.append(
                f"| {row.get('test_set', '')} | {float(row.get('f1', 0.0)):.4f} | {float(row.get('iou', 0.0)):.4f} | {float(row.get('precision', 0.0)):.4f} | {float(row.get('recall', 0.0)):.4f} |"
            )
    else:
        lines.append("Evaluation has not produced summary rows yet.")
    write_text(Path(root) / "RES_SUMMARY.md", "\n".join(lines) + "\n")
