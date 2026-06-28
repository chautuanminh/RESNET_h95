from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from .config import deep_get, load_config, resolve_output_paths
from .datasets import discover_indices, make_dataset
from .gpu import resolve_runtime


def run_smoke(config_path: str | Path, sample_count: int = 2) -> list[dict[str, Any]]:
    config = load_config(config_path)
    paths = resolve_output_paths(config)
    runtime = resolve_runtime(config)
    data_root = deep_get(config, "data.root", "/storage/student7/cmt/data/archive")
    reports: list[dict[str, Any]] = []
    errors: list[str] = []

    print("# ResNet34-H95 Smoke Preflight")
    print(f"cwd: {Path(os.getcwd())}")
    print(f"config_path: {Path(config_path)}")
    print(f"data.root: {data_root}")
    print(f"output.root_dir: {paths['root']}")
    print(f"output.run_dir: {paths['run']}")
    print(
        "runtime: "
        f"device={runtime.get('device')} gpu={runtime.get('gpu_name')} "
        f"amp={runtime.get('amp')} amp_dtype={runtime.get('amp_dtype')}"
    )

    for label, folder in _dataset_specs(config):
        try:
            indices = discover_indices(config, folder)
            report = {"label": label, "folder": folder, "count": len(indices), "samples": []}
            reports.append(report)
            print(f"dataset: label={label} folder={folder} count={len(indices)}")
            if not indices:
                continue
            sample_indices = indices[: max(0, int(sample_count))]
            dataset = make_dataset(config, label, folder, sample_indices)
            try:
                for item in range(len(sample_indices)):
                    sample = dataset[item]
                    stats = _sample_stats(sample)
                    report["samples"].append(stats)
                    print(
                        "  sample: "
                        f"id={stats['image_id']} index={stats['dataset_index']} "
                        f"input_shape={stats['input_shape']} mask_shape={stats['mask_shape']} "
                        f"h95_min={stats['h95_min']:.6f} h95_mean={stats['h95_mean']:.6f} "
                        f"h95_max={stats['h95_max']:.6f} mask_positive_pixels={stats['mask_positive_pixels']} "
                        f"mask_positive_ratio={stats['mask_positive_ratio']:.6f}"
                    )
            finally:
                close = getattr(dataset, "close", None)
                if callable(close):
                    close()
        except Exception as exc:
            message = f"dataset label={label} folder={folder} failed: {exc}"
            errors.append(message)
            print(f"error: {message}")

    if errors:
        raise RuntimeError("Smoke preflight failed:\n" + "\n".join(errors))
    return reports


def _dataset_specs(config: dict[str, Any]) -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    for folder in list(deep_get(config, "data.train_sets", ["DocTamperV1-TrainingSet"])):
        specs.append(("train", str(folder)))
    for label, folder in dict(deep_get(config, "data.test_sets", {})).items():
        specs.append((str(label), str(folder)))
    return specs


def _sample_stats(sample: dict[str, Any]) -> dict[str, Any]:
    import numpy as np  # type: ignore

    h95 = np.asarray(sample.get("h95", sample["input"][1]), dtype="float32")
    mask = np.asarray(sample.get("mask"), dtype="float32")
    positive = mask > 0.5
    positive_pixels = int(positive.sum())
    return {
        "dataset_index": int(sample.get("dataset_index", -1)),
        "image_id": str(sample.get("image_id", "")),
        "input_shape": _shape(sample.get("input")),
        "mask_shape": _shape(sample.get("mask")),
        "h95_min": float(np.nanmin(h95)) if h95.size else 0.0,
        "h95_mean": float(np.nanmean(h95)) if h95.size else 0.0,
        "h95_max": float(np.nanmax(h95)) if h95.size else 0.0,
        "mask_positive_pixels": positive_pixels,
        "mask_positive_ratio": positive_pixels / max(1, int(mask.size)),
    }


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", ())
    return tuple(int(part) for part in shape)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-count", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        run_smoke(args.config, sample_count=args.sample_count)
    except Exception as exc:
        print(f"smoke_failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
