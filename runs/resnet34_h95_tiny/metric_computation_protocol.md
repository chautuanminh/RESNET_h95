# Metric Computation Protocol

- Official threshold metrics at 0.5 use exact streaming TP/FP/FN/TN accumulation.
- Threshold sweep metrics from 0.01 to 0.99 use exact streaming TP/FP/FN/TN accumulation per threshold.
- Per-image metrics are saved as compact scalar rows.
- Full test-set probability arrays are never concatenated or retained in memory.
- AUROC/AUPRC are computed with a bounded histogram approximation when full-pixel exact curves would exceed memory limits.
