from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import as_float


TAMPER_COLUMNS = [
    "tamper_type",
    "forgery_type",
    "manipulation_type",
    "attack_type",
    "category",
    "source_type",
    "label_type",
]


@dataclass(frozen=True)
class TamperTypeAssignment:
    tamper_type: str
    tamper_type_source: str
    tamper_type_confidence: float
    tamper_type_reason: str
    features: dict[str, Any]


def normalize_tamper_label(label: Any) -> str | None:
    if label is None:
        return None
    text = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    mapping = {
        "cm": "copy_move",
        "copy_move": "copy_move",
        "copymove": "copy_move",
        "copy__move": "copy_move",
        "sp": "splicing",
        "splice": "splicing",
        "splicing": "splicing",
        "ge": "generation",
        "gen": "generation",
        "generated": "generation",
        "generation": "generation",
    }
    return mapping.get(text)


def split_aliases(split: str) -> list[str]:
    base = split.replace("DocTamperV1-", "")
    return [split, base, f"DocTamperV1-{base}"]


def load_tamper_metadata(root: str | Path) -> dict[str, dict[int, str]]:
    metadata: dict[str, dict[int, str]] = {}
    source = Path(root)
    if not source.exists():
        return metadata
    for path in sorted(source.glob("DocTamperV1-*.pk")):
        with path.open("rb") as f:
            raw = pickle.load(f)
        if not isinstance(raw, dict):
            continue
        folder = path.stem
        normalized = {
            int(index): normalize_tamper_label(label) or "unknown"
            for index, label in raw.items()
        }
        short = folder.replace("DocTamperV1-", "")
        metadata[folder] = normalized
        metadata[short] = normalized
    return metadata


def lookup_metadata(metadata: dict[str, dict[int, str]], split: str, dataset_index: int) -> str | None:
    for alias in split_aliases(split):
        values = metadata.get(alias)
        if values and int(dataset_index) in values:
            return values[int(dataset_index)]
    return None


def heuristic_tamper_type(features: dict[str, Any]) -> TamperTypeAssignment:
    mask_area_ratio = as_float(features.get("mask_area_ratio"))
    component_count = as_float(features.get("component_count"))
    patch_similarity = as_float(features.get("patch_similarity"))
    h95_ratio = as_float(features.get("h95_inside_outside_ratio"), 1.0)
    edge_ratio = as_float(features.get("edge_inside_outside_ratio"), 1.0)

    if mask_area_ratio <= 0 or mask_area_ratio < as_float(features.get("min_mask_ratio"), 0.00005):
        return TamperTypeAssignment(
            "heuristic_uncertain",
            "heuristic",
            0.2,
            "mask is too small or absent for reliable type inference",
            dict(features),
        )

    if patch_similarity >= 0.85 and component_count >= 1:
        return TamperTypeAssignment(
            "heuristic_copy_move",
            "heuristic",
            min(0.95, 0.55 + (patch_similarity - 0.85) * 2),
            "high same-image patch similarity suggests possible copy-move",
            dict(features),
        )

    if h95_ratio >= 1.35 or edge_ratio >= 1.4:
        return TamperTypeAssignment(
            "heuristic_splicing",
            "heuristic",
            min(0.9, 0.55 + max(h95_ratio - 1.35, edge_ratio - 1.4) * 0.2),
            "local residual or edge statistics differ from surrounding context",
            dict(features),
        )

    if h95_ratio < 0.8 or edge_ratio < 0.8:
        return TamperTypeAssignment(
            "heuristic_generation",
            "heuristic",
            0.62,
            "smooth local statistics and weak duplicate evidence suggest possible generation",
            dict(features),
        )

    return TamperTypeAssignment(
        "heuristic_uncertain",
        "heuristic",
        0.35,
        "signals are weak or conflicting; heuristic label is tentative",
        dict(features),
    )


def assign_tamper_type(
    split: str,
    dataset_index: int,
    metadata: dict[str, dict[int, str]],
    row: dict[str, Any] | None = None,
    enable_heuristic: bool = True,
    heuristic_features: dict[str, Any] | None = None,
) -> TamperTypeAssignment:
    metadata_label = lookup_metadata(metadata, split, int(dataset_index))
    if metadata_label:
        return TamperTypeAssignment(
            metadata_label,
            "metadata",
            1.0,
            "official tampering type metadata from tampering_types pickle",
            dict(heuristic_features or {}),
        )

    row = row or {}
    for column in TAMPER_COLUMNS:
        normalized = normalize_tamper_label(row.get(column))
        if normalized:
            return TamperTypeAssignment(
                normalized,
                "dataset_index",
                1.0,
                f"tamper type read from column {column}",
                dict(heuristic_features or {}),
            )

    if enable_heuristic:
        return heuristic_tamper_type(heuristic_features or {})

    return TamperTypeAssignment("unknown", "unknown", 0.0, "no metadata available and heuristic disabled", {})
