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

Experiment B trains on `data/experiment2/train.csv` (the real training rows plus
681 validated synthetic narratives, every class at least 300 rows) with no class
weighting. Validation and test use `data/experiment2/val.csv` and `test.csv`,
which are identical to the Phase 1 splits. See `data/experiment2/README.md` for
how the synthetic rows were generated and checked.

```bash
python sakina/scripts/run_experiment_b.py
```

Both experiments re-encode every narrative through BERT at up to 512 tokens each
epoch, so use a GPU. `sakina/colab_run.ipynb` runs A and B on a free Colab T4 and
downloads the results as a zip.

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
