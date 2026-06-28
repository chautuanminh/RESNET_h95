# Tampering Type Analysis Report

## Purpose
This diagnostic analysis groups ResNet34-H95 segmentation performance by known or likely tampering type.

## Metadata Warning
Tamper-type analysis depends on metadata availability. If metadata is missing, heuristic labels are not ground truth.

- tamper type sources found: ['metadata']

## Counts And Official Threshold Performance
| test_set | tamper_type | images | f1 | iou | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| TestingSet | generation | 3 | 0.0382 | 0.0195 | 0.0217 | 0.1593 |
| TestingSet | copy_move | 3 | 0.0932 | 0.0489 | 0.0572 | 0.2500 |
| FCD | copy_move | 3 | 0.0410 | 0.0209 | 0.0236 | 0.1566 |
| FCD | generation | 1 | 0.0153 | 0.0077 | 0.0097 | 0.0370 |
| SCD | copy_move | 5 | 0.0629 | 0.0325 | 0.0361 | 0.2417 |

## Diagnostic Thresholds
Best thresholds are analysis-only and do not replace the official 0.5 result.
- best-threshold rows: 5

## Failure Patterns
Failure panels are grouped by test set and tampering type. Each panel includes image index, tamper type, source, confidence, F1, IoU, precision, recall, GT area ratio, predicted area ratio, and error category where available.

## H95 Interpretation
H95 can help when residual signal aligns with the tampered region; it can struggle when compression residuals are weak inside the ground truth or strong outside it.

## Limitations
- if metadata is missing, heuristic labels are not ground truth
- no OCR is used
- copy-move, splicing, and generation may be visually ambiguous
- conclusions from heuristic labels must be treated as tentative
