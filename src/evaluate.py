from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import deep_get, load_config, resolve_output_paths
from .datasets import discover_indices, iter_batches, make_dataset
from .gpu import resolve_runtime
from .metrics import HistogramCurve, StreamingBinaryMetrics, ThresholdSweep, metrics_from_counts, per_image_metric_row
from .models import create_model
from .report import write_pdf_report, write_protocol_documents, write_res_summary
from .utils import ensure_dir, frange, safe_div, write_csv
from .visualization import save_diagnostic_panel, save_placeholder_plot


def run_evaluate(config_path: str | Path, checkpoint: str | None = None) -> list[dict[str, Any]]:
    config = load_config(config_path)
    paths = resolve_output_paths(config)
    root = ensure_dir(paths["root"])
    run_dir = ensure_dir(paths["run"])
    write_protocol_documents(root)
    for sub in ["per_image_metrics", "threshold_sweeps", "examples", "plots"]:
        ensure_dir(run_dir / sub)

    model = create_model(config)
    runtime = resolve_runtime(config)
    _load_checkpoint_if_available(model, checkpoint, config)

    thresholds = frange(
        float(deep_get(config, "evaluation.threshold_min", 0.01)),
        float(deep_get(config, "evaluation.threshold_max", 0.99)),
        float(deep_get(config, "evaluation.threshold_step", 0.01)),
    )
    official_threshold = float(deep_get(config, "evaluation.official_threshold", 0.5))
    batch_size = int(deep_get(config, "evaluation.batch_size", deep_get(config, "training.batch_size", 4)))
    save_examples = int(deep_get(config, "evaluation.save_examples_per_set", 24))

    official_rows: list[dict[str, Any]] = []
    all_per_image_rows: list[dict[str, Any]] = []
    all_sweep_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for label, folder in dict(deep_get(config, "data.test_sets", {})).items():
        indices = discover_indices(config, folder)
        dataset = make_dataset(config, label, folder, indices)
        stream = StreamingBinaryMetrics(official_threshold)
        sweep = ThresholdSweep(thresholds)
        curve = HistogramCurve(bins=int(deep_get(config, "evaluation.histogram_bins", 512)))
        split_per_image: list[dict[str, Any]] = []
        example_count = 0

        for batch in iter_batches(dataset, batch_size):
            probs = _predict_batch(model, batch, config, runtime)
            for sample, prob in zip(batch, probs):
                mask = sample["mask"][0]
                row = per_image_metric_row(prob, mask, official_threshold)
                row.update(
                    {
                        "test_set": label,
                        "dataset_folder": folder,
                        "dataset_index": sample["dataset_index"],
                        "image_id": sample["image_id"],
                        "method": "grayscale + H95",
                        "postprocess": "no_blob",
                        "threshold": official_threshold,
                        "tamper_type": sample.get("tamper_type", "unknown"),
                        "tamper_type_source": sample.get("tamper_type_source", "unknown"),
                        "tamper_type_confidence": sample.get("tamper_type_confidence", 0.0),
                        "tamper_type_reason": sample.get("tamper_type_reason", ""),
                        "h95_mean": _mean(sample.get("h95")),
                    }
                )
                split_per_image.append(row)
                all_per_image_rows.append(row)
                pred = (prob >= official_threshold).astype("uint8")
                if example_count < save_examples:
                    save_diagnostic_panel(
                        run_dir / "examples" / label / f"sample_{int(sample['dataset_index']):06d}_f1_{float(row['f1']):.4f}.png",
                        sample,
                        prob,
                        pred,
                        row,
                        title="Raw no_blob prediction",
                    )
                    example_count += 1
            batch_probs = _stack_probs(probs)
            batch_masks = _stack_masks(batch)
            stream.update(batch_probs, batch_masks)
            sweep.update(batch_probs, batch_masks)
            curve.update(batch_probs, batch_masks)

        counts = stream.counts
        metrics = stream.metrics()
        curve_metrics = curve.approximate()
        official_row = {
            "test_set": label,
            "dataset_folder": folder,
            "method": "grayscale + H95",
            "postprocess": "no_blob",
            "threshold": official_threshold,
            "num_images": len(dataset),
            **counts,
            **metrics,
            **curve_metrics,
            "positive_gt_pixels": counts["tp"] + counts["fn"],
            "predicted_positive_pixels": counts["tp"] + counts["fp"],
        }
        official_rows.append(official_row)
        split_sweep = sweep.rows(
            {"test_set": label, "dataset_folder": folder, "method": "grayscale + H95", "postprocess": "no_blob"}
        )
        all_sweep_rows.extend(split_sweep)
        best = sweep.best_by_f1()
        best_rows.append({**best, "test_set": label, "analysis_only": True})
        write_csv(run_dir / "per_image_metrics" / f"{label}_resnet34_h95_no_blob.csv", split_per_image)
        write_csv(run_dir / "threshold_sweeps" / f"{label}_threshold_sweep.csv", split_sweep)

    write_csv(run_dir / "official_threshold_0.5_metrics.csv", official_rows)
    write_csv(run_dir / "summary_resnet34_h95_trained.csv", official_rows)
    write_csv(run_dir / "best_threshold_metrics.csv", best_rows)
    write_csv(run_dir / "threshold_sweep.csv", all_sweep_rows)
    write_csv(run_dir / "per_image_metrics.csv", all_per_image_rows)
    _write_plots(run_dir / "plots")
    write_res_summary(root, official_rows)
    write_pdf_report(root, "Evaluation completed with streaming official and diagnostic metrics.")
    return official_rows


def _predict_batch(model, batch: list[dict[str, Any]], config: dict[str, Any], runtime: dict[str, Any]):
    import numpy as np  # type: ignore

    if bool(deep_get(config, "runtime.dummy", False)) or model.__class__.__name__ == "DummyResNetUNet":
        return [np.clip(sample["input"][1], 0.0, 1.0).astype("float32") for sample in batch]
    try:
        import torch  # type: ignore

        device = torch.device(runtime["device"])
        if not next(model.parameters()).is_cuda and runtime["device"] == "cuda":
            model.to(device)
        model.eval()
        x = torch.from_numpy(np.stack([sample["input"] for sample in batch])).to(device)
        if runtime["device"] == "cuda" and runtime.get("channels_last"):
            x = x.contiguous(memory_format=torch.channels_last)
        amp_dtype = torch.bfloat16 if runtime.get("amp_dtype") == "bfloat16" else torch.float16
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=runtime["device"] == "cuda"):
                logits = model(x)
            probs = torch.sigmoid(logits).float().detach().cpu().numpy()[:, 0]
        return [probs[i] for i in range(probs.shape[0])]
    except Exception as exc:
        raise RuntimeError(f"Model inference failed: {exc}") from exc


def _load_checkpoint_if_available(model, checkpoint: str | None, config: dict[str, Any]) -> None:
    if not checkpoint or bool(deep_get(config, "runtime.dummy", False)) or model.__class__.__name__ == "DummyResNetUNet":
        return
    try:
        import torch  # type: ignore

        payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("model_state", payload)
        state = {key.replace("_orig_mod.", ""): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise RuntimeError(f"Unable to load checkpoint {checkpoint}: {exc}") from exc


def _mean(array) -> float:
    try:
        import numpy as np  # type: ignore

        return float(np.asarray(array).mean())
    except Exception:
        return 0.0


def _stack_probs(probs):
    import numpy as np  # type: ignore

    return np.stack(probs)


def _stack_masks(batch):
    import numpy as np  # type: ignore

    return np.stack([sample["mask"][0] for sample in batch])


def _write_plots(plot_dir: Path) -> None:
    for name in [
        "f1_resnet34_h95_by_testset.png",
        "iou_resnet34_h95_by_testset.png",
        "precision_resnet34_h95_by_testset.png",
        "recall_resnet34_h95_by_testset.png",
        "f1_resnet34_vs_comparison_md.png",
        "iou_resnet34_vs_comparison_md.png",
    ]:
        save_placeholder_plot(plot_dir / name, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    run_evaluate(args.config, args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
