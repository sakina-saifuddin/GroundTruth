# Muhammad's BiLSTM-Attention classifier

This folder contains Phase 2, Muhammad's model for predicting one of eight PHMSA
gas-pipeline incident causes from a cleaned narrative. A Bidirectional LSTM (BiLSTM)
reads word embeddings trained from scratch, and an additive attention mechanism pools
the hidden states into a single representation for classification.

Unlike Sakina's frozen-BERT approach, all parameters here are trainable — including
the embedding layer. The attention mechanism learns to highlight the words most relevant
to the predicted cause.

## Architecture

```
narrative text
   │
   ▼
Word tokenization + vocabulary lookup
   │
   ▼
nn.Embedding(vocab_size, embed_dim=128)
   │
   ▼
nn.LSTM(bidirectional=True, hidden_size=128, num_layers=2)   → (batch, seq, 256)
   │
   ▼
AdditiveAttention (learnable query, padding-masked softmax)   → (batch, 256)
   │
   ▼
nn.Dropout(0.3) → nn.Linear(256, 8)
   │
   ▼
cause logits
```

## Setup

From the repository root, install the model dependencies:

```bash
python -m pip install -r muhammad/requirements.txt
```

The prepared files must already exist under `data/processed/` (Experiment A) and
`data/experiment2/` (Experiment B). No external model weights are downloaded — the
vocabulary and embeddings are built from the training data.

## Run the experiments

Experiment A uses the original clean training split with class weights:

```bash
python muhammad/scripts/run_experiment_a.py
```

Experiment B trains on `data/experiment2/train.csv` (real rows plus validated synthetic
narratives, every class at least 300 rows) with no class weighting:

```bash
python muhammad/scripts/run_experiment_b.py
```

A GPU is used automatically when available. CPU training is slower but works fine since
the model is much lighter than a transformer.

## Results

Each completed run writes the following under its experiment directory:

- `metrics.json`: measured accuracy, macro/micro F1, per-class precision, recall, F1
  and support, confusion matrix, best validation macro-F1, best epoch, trainable
  parameter count, and training history.
- `per_class_report.txt`: readable test-set metrics by cause.
- `confusion_matrix.png`: confusion matrix normalized by true class.

`results/comparison.md` is refreshed after each experiment. Until both runs have
results, missing values are marked pending and no ranking is claimed.
Checkpoints and vocabulary files are saved under `checkpoints/` (excluded from Git).

## Colab

`colab_run.ipynb` runs both experiments on a free Colab T4 GPU and saves results to
Google Drive. Set the runtime to T4 GPU before running.

## Course concepts

- **Lecture 7 — Recurrent Neural Networks:** the BiLSTM captures sequential context
  in both directions across the narrative tokens.
- **Lecture 8 — Attention:** additive attention pools the BiLSTM outputs into a single
  attended representation, weighting words by their relevance to the predicted cause.
- **Lecture 9 — Cross-Entropy Loss:** multiclass training uses `CrossEntropyLoss`;
  Experiment A supplies training-only class weights.
- **Lecture 12 — LLMs:** Experiment B evaluates separately supplied, AI-assisted
  synthetic training narratives. Synthetic text is never taken from validation or test.
