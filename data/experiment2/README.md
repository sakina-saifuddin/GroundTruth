# Experiment 2 — AI-Assisted Smart Oversampling

Training data: `train.csv` is pending; it will contain real and synthetic
narratives after the separate synthetic-data generation step. Validation and
test files are copies of the Phase 1 splits and must remain unchanged.

How to handle imbalance: **do not use class weights**. Use plain
`torch.nn.CrossEntropyLoss()` with no weight argument.

Synthetic narratives are planned to be generated using the Gemini API for rare
classes only:

- Material Failure of Pipe or Weld: 40 synthetic rows needed.
- Excavation Damage: 47 synthetic rows needed.
- Natural Force Damage: 89 synthetic rows needed.
- Other Outside Force Damage: 153 synthetic rows needed.
- Other Incident Cause: 167 synthetic rows needed.
- Incorrect Operation: 185 synthetic rows needed.

The target is at least 300 training examples per class. These counts are based
on the current Phase 1 training split. Synthetic narratives must be added to
training data only; never use validation or test text to create synthetic
examples.

Files:

- `train.csv`: **PENDING** — to be added after synthetic generation.
- `val.csv`: 456 rows; same as Experiment 1.
- `test.csv`: 458 rows; same as Experiment 1.

The split and oversampling workflow uses random seed 42.
