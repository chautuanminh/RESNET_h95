from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .utils import flatten_numeric, safe_div


COUNT_KEYS = ["tp", "fp", "fn", "tn"]


def empty_counts() -> dict[str, int]:
    return {key: 0 for key in COUNT_KEYS}


def confusion_from_flat(probs: Iterable[float], labels: Iterable[float], threshold: float = 0.5) -> dict[str, int]:
    try:
        import numpy as np  # type: ignore

        probs_array = np.asarray(probs)
        labels_array = np.asarray(labels)
        if probs_array.shape == labels_array.shape and probs_array.size > 0:
            pred = probs_array >= threshold
            gt = labels_array >= 0.5
            return {
                "tp": int(np.count_nonzero(pred & gt)),
                "fp": int(np.count_nonzero(pred & ~gt)),
                "fn": int(np.count_nonzero(~pred & gt)),
                "tn": int(np.count_nonzero(~pred & ~gt)),
            }
    except Exception:
        pass
    counts = empty_counts()
    for prob, label in zip(flatten_numeric(list(probs)), flatten_numeric(list(labels))):
        pred = prob >= threshold
        gt = label >= 0.5
        if pred and gt:
            counts["tp"] += 1
        elif pred and not gt:
            counts["fp"] += 1
        elif not pred and gt:
            counts["fn"] += 1
        else:
            counts["tn"] += 1
    return counts


def add_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    for key in COUNT_KEYS:
        left[key] = int(left.get(key, 0)) + int(right.get(key, 0))
    return left


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    tn = int(counts.get("tn", 0))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    iou = safe_div(tp, tp + fp + fn)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "accuracy": accuracy,
        "specificity": specificity,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
    }


@dataclass
class StreamingBinaryMetrics:
    threshold: float = 0.5
    counts: dict[str, int] = field(default_factory=empty_counts)

    def update(self, probs, labels) -> None:
        add_counts(self.counts, confusion_from_flat(flatten_numeric(probs), flatten_numeric(labels), self.threshold))

    def metrics(self) -> dict[str, float]:
        return metrics_from_counts(self.counts)


@dataclass
class ThresholdSweep:
    thresholds: list[float]
    counts_by_threshold: dict[float, dict[str, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.thresholds = [round(float(value), 6) for value in self.thresholds]
        self.counts_by_threshold = {threshold: empty_counts() for threshold in self.thresholds}

    def update(self, probs, labels) -> None:
        flat_probs = flatten_numeric(probs)
        flat_labels = flatten_numeric(labels)
        for threshold in self.thresholds:
            add_counts(self.counts_by_threshold[threshold], confusion_from_flat(flat_probs, flat_labels, threshold))

    def rows(self, prefix: dict | None = None) -> list[dict]:
        rows = []
        for threshold in self.thresholds:
            counts = self.counts_by_threshold[threshold]
            row = dict(prefix or {})
            row.update({"threshold": threshold, **counts, **metrics_from_counts(counts)})
            rows.append(row)
        return rows

    def best_by_f1(self) -> dict:
        rows = self.rows()
        return max(rows, key=lambda row: (row["f1"], row["iou"], -row["threshold"])) if rows else {}


class HistogramCurve:
    """Bounded-memory approximate AUROC/AUPRC accumulator."""

    def __init__(self, bins: int = 1000) -> None:
        self.bins = bins
        self.pos = [0] * bins
        self.neg = [0] * bins

    def update(self, probs, labels) -> None:
        for prob, label in zip(flatten_numeric(probs), flatten_numeric(labels)):
            index = min(self.bins - 1, max(0, int(float(prob) * self.bins)))
            if float(label) >= 0.5:
                self.pos[index] += 1
            else:
                self.neg[index] += 1

    def approximate(self) -> dict[str, float]:
        total_pos = sum(self.pos)
        total_neg = sum(self.neg)
        if total_pos == 0 or total_neg == 0:
            return {"auroc_approx": 0.0, "auprc_approx": 0.0}
        tp = fp = 0
        roc_points = [(0.0, 0.0)]
        pr_points = []
        for index in range(self.bins - 1, -1, -1):
            tp += self.pos[index]
            fp += self.neg[index]
            recall = safe_div(tp, total_pos)
            fpr = safe_div(fp, total_neg)
            precision = safe_div(tp, tp + fp)
            roc_points.append((fpr, recall))
            pr_points.append((recall, precision))
        auroc = _trapz(roc_points)
        auprc = _trapz(sorted(pr_points))
        return {"auroc_approx": auroc, "auprc_approx": auprc}


def _trapz(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2
    return max(0.0, min(1.0, area))


def per_image_metric_row(probs, labels, threshold: float = 0.5) -> dict:
    counts = confusion_from_flat(flatten_numeric(probs), flatten_numeric(labels), threshold)
    metrics = metrics_from_counts(counts)
    total = sum(counts.values())
    return {
        **counts,
        **metrics,
        "gt_area_ratio": safe_div(counts["tp"] + counts["fn"], total),
        "pred_area_ratio": safe_div(counts["tp"] + counts["fp"], total),
        "error_area_ratio": safe_div(counts["fp"] + counts["fn"], total),
    }
