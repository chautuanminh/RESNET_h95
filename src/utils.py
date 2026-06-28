from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_writable_dir(path: str | Path, purpose: str = "output") -> Path:
    resolved = Path(path)
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        raise PermissionError(_writable_dir_error(resolved, purpose, "could not create it")) from exc
    except OSError as exc:
        raise OSError(f"Could not create {purpose} directory: {resolved}. Original error: {exc}") from exc
    if not resolved.is_dir():
        raise NotADirectoryError(f"{purpose} path exists but is not a directory: {resolved}")
    if not os.access(resolved, os.W_OK):
        raise PermissionError(_writable_dir_error(resolved, purpose, "directory is not writable"))
    return resolved


def _writable_dir_error(path: Path, purpose: str, reason: str) -> str:
    return (
        f"{purpose} directory is not writable: {path} ({reason}). "
        "Diagnose on the server with: pwd; whoami; ls -ld res; "
        "ls -ld res/doctamper_resnet34_h95_35epochs_comparison; "
        "ls -l res/doctamper_resnet34_h95_35epochs_comparison/training.log. "
        "Prefer changing the config output.root_dir to a fresh writable folder under runs/. "
        "If you own the directory or file, use chmod u+w on that path. "
        "Do not delete or overwrite existing results unless the folder is confirmed disposable."
    )


def write_text(path: str | Path, text: str) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
    return target


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    materialized = list(rows)
    if fieldnames is None:
        fieldnames = []
        for row in materialized:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with target.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return target


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_int_csv(path: str | Path, column: str = "dataset_index") -> list[int]:
    return [int(row[column]) for row in read_csv_rows(path)]


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def frange(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + step / 10:
        values.append(round(current, 6))
        current += step
    return values


def flatten_numeric(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        out: list[float] = []
        for value in values:
            out.extend(flatten_numeric(value))
        return out
    shape = getattr(values, "shape", None)
    if shape is not None:
        try:
            return [float(x) for x in values.reshape(-1).tolist()]
        except Exception:
            pass
    try:
        return [float(values)]
    except (TypeError, ValueError):
        return []


def write_minimal_pdf(path: str | Path, title: str, body: str) -> Path:
    target = Path(path)
    ensure_dir(target.parent)
    title = title.replace("(", "[").replace(")", "]")
    body = body.replace("(", "[").replace(")", "]")
    content = f"BT /F1 16 Tf 72 760 Td ({title}) Tj /F1 10 Tf 0 -28 Td ({body[:600]}) Tj ET"
    stream = content.encode("latin-1", errors="replace")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode("ascii") + stream + b"\nendstream endobj\n",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode("ascii")
    )
    target.write_bytes(bytes(pdf))
    return target


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
