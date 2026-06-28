# Leakage Check Report

## Folder Counts
- train folder: DocTamperV1-TrainingSet
- train count: 24
- validation count: 6
- final evaluation folder TestingSet: 6
- final evaluation folder FCD: 4
- final evaluation folder SCD: 5

## Proofs
- TestingSet/FCD/SCD are only constructed by evaluation loaders.
- train/validation image key overlap count: 0
- model input is constructed from grayscale image and H95 only; masks are returned as targets.
- checkpoint selection policy: best checkpoint selected from internal validation metrics only
