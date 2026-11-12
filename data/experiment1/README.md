# Experiment 1 — Class-Weighted Loss

Training data: `train.csv` (original real narratives only; the classes are imbalanced).
Validation and test data are unchanged from Phase 1.

How to handle imbalance: use `class_weights.json` in the loss function. The JSON
contains an `ordered_by_label` list for use with
`torch.nn.CrossEntropyLoss(weight=torch.tensor(ordered_by_label))`.

Do **not** use class weights in Experiment 2.

Files:

- `train.csv`: 2,130 rows, 8 classes, imbalanced.
- `val.csv`: 456 rows.
- `test.csv`: 458 rows.
- `class_weights.json`: weights computed from the Phase 1 training split only.

The split uses random seed 42. The byte-for-byte copies in this folder are
verified against their Phase 1 source files.
