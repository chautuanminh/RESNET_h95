from __future__ import annotations

import argparse
from pathlib import Path

from .config import deep_get, load_config, resolve_output_paths
from .splits import FINAL_EVAL_DATASETS
from .utils import ensure_dir, read_csv_rows, read_int_csv, write_text


def _forbidden_token() -> str:
    return "blob" + "_minarea"


def validate_no_forbidden_artifacts(root: str | Path) -> tuple[bool, list[str]]:
    token = _forbidden_token()
    errors: list[str] = []
    root_path = Path(root)
    if not root_path.exists():
        return True, []
    for path in root_path.rglob("*"):
        if path.is_dir():
            continue
        if token in path.name:
            errors.append(f"Forbidden artifact name: {path}")
            continue
        if path.suffix.lower() in {".py", ".yaml", ".yml", ".json", ".md", ".csv", ".txt"}:
            try:
                if token in path.read_text(encoding="utf-8", errors="ignore"):
                    errors.append(f"Forbidden artifact text: {path}")
            except OSError:
                pass
    return not errors, errors


def validate_required_outputs(root: str | Path, require_tamper_outputs: bool = True) -> tuple[bool, list[str]]:
    root_path = Path(root)
    required = [
        "PLAN.md",
        "RES_SUMMARY.md",
        "output.md",
        "resnet34_h95_evaluation_report.pdf",
        "model_summary.txt",
        "preprocessing_protocol.md",
        "metric_computation_protocol.md",
        "threshold_policy.md",
        "leakage_check_report.md",
    ]
    errors = [f"Missing required root output: {name}" for name in required if not (root_path / name).exists()]
    run_dir = root_path / "doctamper_resnet34_h95_35epochs_comparison"
    for name in [
        "config_snapshot.yaml",
        "config_resolved.yaml",
        "split_summary.md",
        "dataset_index.csv",
        "train_metrics.csv",
        "val_metrics.csv",
        "summary_resnet34_h95_trained.csv",
        "official_threshold_0.5_metrics.csv",
        "best_threshold_metrics.csv",
        "threshold_sweep.csv",
        "per_image_metrics.csv",
        "batch_size_autotune.csv",
        "gpu_profile.txt",
        "plots/training_curves.png",
        "checkpoints/best_model.pth",
        "checkpoints/last_checkpoint.pth",
        "splits/train_indices_seed42.csv",
        "splits/val_indices_seed42.csv",
    ]:
        if not (run_dir / name).exists():
            errors.append(f"Missing required run output: {name}")
    if require_tamper_outputs:
        failure_dir = root_path / "failure_case_analysis"
        for name in [
            "failure_analysis_config.yaml",
            "failure_analysis_config.json",
            "failure_case_report.md",
            "failure_cases_all_selected.csv",
            "failure_summary_by_test_set.csv",
            "failure_summary_by_primary_category.csv",
            "failure_summary_by_raw_category.csv",
            "failure_summary_by_severity.csv",
            "failure_summary_by_likely_reason.csv",
            "failure_summary_by_category.csv",
            "failure_summary_by_likely_tamper_type.csv",
            "threshold_sweep_0.01_0.99.csv",
        ]:
            if not (failure_dir / name).exists():
                errors.append(f"Missing required failure output: {name}")
        tamper_dir = root_path / "tampering_type_analysis"
        for name in [
            "tampering_type_analysis_config.yaml",
            "tampering_type_analysis_report.md",
            "tampering_type_summary_by_test_set.csv",
            "tampering_type_summary_by_type.csv",
            "tampering_type_summary_by_test_set_and_type.csv",
            "tampering_type_per_image.csv",
            "tampering_type_threshold_sweep.csv",
            "tampering_type_best_thresholds.csv",
            "tampering_type_failure_summary.csv",
            "tampering_type_feature_summary.csv",
        ]:
            if not (tamper_dir / name).exists():
                errors.append(f"Missing required tamper output: {name}")
    return not errors, errors


def validate_experiment(config_path: str | Path) -> tuple[bool, list[str]]:
    config = load_config(config_path)
    paths = resolve_output_paths(config)
    root = paths["root"]
    ensure_dir(root)
    errors: list[str] = []

    train_sets = set(deep_get(config, "data.train_sets", ["DocTamperV1-TrainingSet"]))
    val_sets = set(deep_get(config, "data.val_sets", []))
    bad = (train_sets | val_sets) & FINAL_EVAL_DATASETS
    if bad:
        errors.append(f"Official evaluation folders configured for train/validation: {sorted(bad)}")

    if int(deep_get(config, "model.in_channels", 2)) != 2:
        errors.append("Model input channel count must be exactly 2.")

    train_csv = paths["run"] / "splits" / "train_indices_seed42.csv"
    val_csv = paths["run"] / "splits" / "val_indices_seed42.csv"
    if train_csv.exists() and val_csv.exists():
        overlap = set(read_int_csv(train_csv)) & set(read_int_csv(val_csv))
        if overlap:
            errors.append(f"Train/validation split overlap: {len(overlap)}")

    for scan_root in [paths["root"], paths["run"], paths["failure"], paths["tamper"], Path("src"), Path("configs")]:
        ok, scan_errors = validate_no_forbidden_artifacts(scan_root)
        if not ok:
            errors.extend(scan_errors)

    require_setting = deep_get(config, "validation.require_outputs", True)
    if str(require_setting).lower() == "auto":
        require_outputs = (paths["run"] / "official_threshold_0.5_metrics.csv").exists()
    else:
        require_outputs = bool(require_setting)
    if require_outputs:
        ok, output_errors = validate_required_outputs(
            root,
            require_tamper_outputs=bool(deep_get(config, "tampering_type_analysis.enabled", True)),
        )
        if not ok:
            errors.extend(output_errors)

    tamper_per_image = paths["tamper"] / "tampering_type_per_image.csv"
    if tamper_per_image.exists():
        for i, row in enumerate(read_csv_rows(tamper_per_image), start=2):
            if not row.get("tamper_type"):
                errors.append(f"tampering_type_per_image.csv row {i} has no tamper_type")
            if row.get("tamper_type_source") == "heuristic" and (
                not row.get("tamper_type_confidence") or not row.get("tamper_type_reason")
            ):
                errors.append(f"heuristic row {i} lacks confidence or reason")
    report = paths["tamper"] / "tampering_type_analysis_report.md"
    if report.exists():
        text = report.read_text(encoding="utf-8", errors="ignore").lower()
        if "heuristic labels are not ground truth" not in text:
            errors.append("tampering type report lacks heuristic limitation warning")

    lines = ["# Validate Experiment Report", "", f"- status: {'PASS' if not errors else 'FAIL'}"]
    if errors:
        lines.append("")
        lines.append("## Errors")
        lines.extend(f"- {error}" for error in errors)
    else:
        lines.extend(
            [
                "- official evaluation folders are excluded from train/validation",
                "- model input channel count is exactly 2",
                "- no forbidden minimum-area artifacts were found in generated code/config/output scan roots",
                "- any no_blob naming is a raw thresholded sigmoid compatibility label",
            ]
        )
    write_text(root / "validate_experiment_report.md", "\n".join(lines) + "\n")
    return not errors, errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    ok, errors = validate_experiment(args.config)
    if errors:
        for error in errors:
            print(error)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
