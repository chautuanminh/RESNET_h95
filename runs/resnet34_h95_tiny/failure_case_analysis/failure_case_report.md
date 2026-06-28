# Failure Case Analysis Report

## 1. Scope
- model name: ResNet34-H95
- checkpoint path: runs\resnet34_h95_tiny\doctamper_resnet34_h95_35epochs_comparison\checkpoints\best_model.pth
- test sets: TestingSet, FCD, SCD
- selected images: 6
- selection rule: f1 ascending, iou ascending, dataset_index ascending
- threshold sweep range: 0.01-0.99 step 0.01

## 2. Summary By Test Set
| test_set | selected_images | mean_f1 | mean_iou | mean_precision | mean_recall | mean_h95_signal_ratio |
| --- | --- | --- | --- | --- | --- | --- |
| FCD | 2 | 0.0227 | 0.0115 | 0.0127 | 0.1810 | 1.2403 |
| SCD | 2 | 0.0227 | 0.0115 | 0.0127 | 0.1810 | 1.2403 |
| TestingSet | 2 | 0.0227 | 0.0115 | 0.0127 | 0.1810 | 1.2403 |

## 3. Severity Breakdown
| test_set | severity | selected_images | mean_f1 | mean_iou |
| --- | --- | --- | --- | --- |
| FCD | catastrophic | 2 | 0.0227 | 0.0115 |
| SCD | catastrophic | 2 | 0.0227 | 0.0115 |
| TestingSet | catastrophic | 2 | 0.0227 | 0.0115 |

## 4. Primary Failure Categories
| test_set | primary_failure_category | severity | selected_images | mean_f1 | mean_precision | mean_recall |
| --- | --- | --- | --- | --- | --- | --- |
| FCD | both_fp_and_fn | catastrophic | 2 | 0.0227 | 0.0127 | 0.1810 |
| SCD | both_fp_and_fn | catastrophic | 2 | 0.0227 | 0.0127 | 0.1810 |
| TestingSet | both_fp_and_fn | catastrophic | 2 | 0.0227 | 0.0127 | 0.1810 |

## 5. Raw Pixel-Error Categories
| test_set | raw_pixel_category | severity | selected_images | mean_f1 |
| --- | --- | --- | --- | --- |
| FCD | false_positive_heavy | catastrophic | 2 | 0.0227 |
| SCD | false_positive_heavy | catastrophic | 2 | 0.0227 |
| TestingSet | false_positive_heavy | catastrophic | 2 | 0.0227 |

## 6. H95 Diagnostic Findings
- H95 available rows: 6
- weak H95 forensic signal rows: 3
- strong H95 forensic signal rows: 0
- mean H95 signal ratio: 1.2403

## 7. Calibration Findings
- threshold sweep rows: 594
- strong calibration failures: 0
- mean threshold F1 gap: 0.0064

## 8. Most Important Failure Modes
| test_set | likely_reason | selected_images | mean_f1 |
| --- | --- | --- | --- |
| FCD | fragmented_prediction_noise | 2 | 0.0227 |
| FCD | large_over_prediction | 2 | 0.0227 |
| FCD | tiny_tamper_region | 2 | 0.0227 |
| FCD | weak_h95_forensic_signal | 1 | 0.0153 |
| SCD | fragmented_prediction_noise | 2 | 0.0227 |
| SCD | large_over_prediction | 2 | 0.0227 |
| SCD | tiny_tamper_region | 2 | 0.0227 |
| SCD | weak_h95_forensic_signal | 1 | 0.0153 |
| TestingSet | fragmented_prediction_noise | 2 | 0.0227 |
| TestingSet | large_over_prediction | 2 | 0.0227 |
| TestingSet | tiny_tamper_region | 2 | 0.0227 |
| TestingSet | weak_h95_forensic_signal | 1 | 0.0153 |

## 9. Recommended Improvements
- calibrate threshold per deployment domain when threshold gaps are consistently positive
- prioritize false-positive suppression for over-detection categories
- improve tiny-region sensitivity where tiny tamper regions dominate likely reasons
- inspect SCD and TestingSet domain-specific failures separately before changing training data

## 10. Artifact List
- failure_cases_all_selected.csv
- threshold_sweep_0.01_0.99.csv
- failure_summary_by_test_set.csv
- failure_summary_by_primary_category.csv
- failure_summary_by_raw_category.csv
- failure_summary_by_severity.csv
- failure_summary_by_likely_reason.csv
- selected_worst_200/<test_set>/rank_*.png
- plots/*.png

## 11. Visualization Artifacts
- plots/worst200_f1_distribution.png
- plots/failure_category_counts.png
- plots/severity_counts_by_test_set.png
- plots/raw_category_counts.png
- plots/precision_recall_scatter.png
- plots/threshold_gap_distribution.png
- plots/pred_gt_area_ratio_vs_f1.png
- plots/h95_signal_ratio_vs_f1.png
- plots/confusion_totals_by_test_set.png
