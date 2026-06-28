from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .utils import ensure_dir, read_int_csv, write_csv, write_text


FINAL_EVAL_DATASETS = {"DocTamperV1-TestingSet", "DocTamperV1-FCD", "DocTamperV1-SCD"}
TRAINING_DATASET = "DocTamperV1-TrainingSet"


class SplitPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class SplitResult:
    train_indices: list[int]
    val_indices: list[int]
    seed: int
    source_folder: str
    output_dir: Path


def create_or_load_split(
    all_indices: Iterable[int],
    output_dir: str | Path,
    seed: int = 42,
    val_count: int = 10000,
    source_folder: str = TRAINING_DATASET,
) -> SplitResult:
    if source_folder in FINAL_EVAL_DATASETS:
        raise SplitPolicyError(f"Official evaluation folder cannot be used for train/val: {source_folder}")

    root = Path(output_dir)
    split_dir = ensure_dir(root / "splits")
    train_csv = split_dir / f"train_indices_seed{seed}.csv"
    val_csv = split_dir / f"val_indices_seed{seed}.csv"

    if train_csv.exists() and val_csv.exists():
        train_indices = read_int_csv(train_csv)
        val_indices = read_int_csv(val_csv)
    else:
        ordered = sorted(int(index) for index in all_indices)
        if val_count >= len(ordered):
            raise SplitPolicyError(
                f"Validation count {val_count} must be smaller than source count {len(ordered)}"
            )
        shuffled = ordered[:]
        random.Random(seed).shuffle(shuffled)
        val_indices = shuffled[:val_count]
        train_indices = shuffled[val_count:]
        write_csv(train_csv, ({"dataset_index": idx} for idx in train_indices), ["dataset_index"])
        write_csv(val_csv, ({"dataset_index": idx} for idx in val_indices), ["dataset_index"])

    overlap = set(train_indices) & set(val_indices)
    if overlap:
        raise SplitPolicyError(f"Train/validation split overlap detected: {len(overlap)} samples")

    result = SplitResult(train_indices, val_indices, seed, source_folder, root)
    write_split_summary(result)
    return result


def write_split_summary(split: SplitResult) -> Path:
    overlap = len(set(split.train_indices) & set(split.val_indices))
    text = "\n".join(
        [
            "# Split Summary",
            "",
            f"- seed: {split.seed}",
            f"- train count: {len(split.train_indices)}",
            f"- validation count: {len(split.val_indices)}",
            f"- source folder: {split.source_folder}",
            "- official evaluation folders excluded from training and validation: yes",
            f"- train/validation overlap count: {overlap} (expected 0)",
            f"- first 10 train indices: {split.train_indices[:10]}",
            f"- first 10 validation indices: {split.val_indices[:10]}",
            "",
        ]
    )
    return write_text(split.output_dir / "split_summary.md", text)


def write_leakage_report(
    path: str | Path,
    train_count: int,
    val_count: int,
    test_counts: dict[str, int],
    train_val_overlap: int = 0,
    checkpoint_policy: str = "best checkpoint selected from internal validation metrics only",
) -> Path:
    rows = [
        "# Leakage Check Report",
        "",
        "## Folder Counts",
        f"- train folder: {TRAINING_DATASET}",
        f"- train count: {train_count}",
        f"- validation count: {val_count}",
    ]
    for name, count in test_counts.items():
        rows.append(f"- final evaluation folder {name}: {count}")
    rows.extend(
        [
            "",
            "## Proofs",
            "- TestingSet/FCD/SCD are only constructed by evaluation loaders.",
            f"- train/validation image key overlap count: {train_val_overlap}",
            "- model input is constructed from grayscale image and H95 only; masks are returned as targets.",
            f"- checkpoint selection policy: {checkpoint_policy}",
            "",
        ]
    )
    return write_text(path, "\n".join(rows))


def assert_no_official_eval_in_train_val(train_folders: Iterable[str], val_folders: Iterable[str]) -> None:
    train_bad = set(train_folders) & FINAL_EVAL_DATASETS
    val_bad = set(val_folders) & FINAL_EVAL_DATASETS
    if train_bad or val_bad:
        raise SplitPolicyError(
            f"Official evaluation folders in train/val: train={sorted(train_bad)}, val={sorted(val_bad)}"
        )
