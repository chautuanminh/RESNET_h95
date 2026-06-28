# Tampering Type Analysis Summary

- Tamper types are obtained from official metadata pickles when available, then existing metric columns, then non-OCR heuristics, then unknown.
- easiest type by official F1: copy_move
- hardest type by official F1: generation
- Failure patterns are grouped in the tampering type analysis folder.
- These diagnostics indicate where the grayscale + H95 residual method aligns or struggles by manipulation type.
