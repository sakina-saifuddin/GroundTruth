# Sakina's frozen-BERT classifier

This folder contains Phase 2, Sakina's model for predicting one of eight PHMSA
gas-pipeline incident causes from a cleaned narrative. BERT-base-uncased is
used as a frozen text encoder. A single linear layer reads its pooled `[CLS]`
representation and predicts the cause. This is multinomial logistic
regression on top of fixed BERT features.

All BERT parameters are frozen. Only the linear head is trained:
`768 × 8 + 8 = 6,152` trainable parameters.

## Setup

From the repository root, install the model dependencies:

```bash
python -m pip install -r sakina/requirements.txt
```

The prepared files must already exist under `data/processed/`. The first run
downloads `bert-base-uncased` from Hugging Face if it is not cached. A CUDA GPU
is used when available; otherwise training runs on CPU and can take a long
time because every narrative is encoded by BERT at up to 512 tokens.

## Run the experiments

Experiment A uses the original training split with the Phase 1 class weights:

```bash
python sakina/scripts/run_experiment_a.py
```

Experiment B uses extra synthetic narratives without class weighting. It
requires a separately reviewed `data/processed/synthetic_train.csv` containing
`narrative_clean`, `CAUSE`, and `label`; this project does not fabricate or
auto-generate those narratives. The 300-example rule identifies six training
causes below the threshold in the verified Phase 1 split, so the synthetic file
must bring all six to at least 300. The current Phase 1 counts include
EXCAVATION DAMAGE (251), in addition to the five classes named in the
generation note. The augmented output is written to
`sakina/processed/augmented_train.csv` to keep Phase 1 files unchanged.

```bash
python sakina/scripts/run_experiment_b.py
```

The oversampling script checks label-map consistency, duplicate narratives,
and overlap with validation and test before writing the augmented training set.
It raises a clear error if the synthetic file is missing or fails a check.

## Results

Each completed run writes the following under its experiment directory:

- `metrics.json`: measured accuracy, macro/micro F1, per-class precision,
  recall, F1 and support, confusion matrix, best validation macro-F1, best
  epoch, and training history.
- `per_class_report.txt`: readable test-set metrics by cause.
- `confusion_matrix.png`: confusion matrix normalized by true class, so each
  row shows recall proportions.

`results/comparison.md` is refreshed after each experiment. Until both runs
have results, missing values are marked pending and no ranking is claimed.
Model checkpoints are saved under `checkpoints/` and are excluded from Git.

## Course concepts

- **Lecture 6 — Logistic Regression:** the linear head classifies fixed BERT
  representations.
- **Lecture 9 — Cross-Entropy Loss:** multiclass training uses
  `CrossEntropyLoss`; Experiment A supplies training-only class weights.
- **Lecture 13 — Masked Language Models / BERT:** BERT-base-uncased supplies
  contextual token representations, with its weights frozen.
- **Lecture 12 — LLMs:** Experiment B evaluates separately supplied,
  AI-assisted synthetic training narratives. Synthetic text is never taken
  from validation or test.
