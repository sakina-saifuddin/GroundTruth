# Experiment 2 — AI-Assisted Smart Oversampling

Training data: `train.csv` is the real Experiment 1 training split plus 681
validated synthetic narratives. Validation and test files are copies of the
Phase 1 splits and are unchanged.

How to handle imbalance: **do not use class weights**. Use plain
`torch.nn.CrossEntropyLoss()` with no weight argument.

## Files

- `train.csv`: **2,811 rows** (2,130 real + 681 synthetic). A `source` column
  marks each row as `real` or `synthetic`.
- `val.csv`: 456 rows; same as Experiment 1.
- `test.csv`: 458 rows; same as Experiment 1.

Class distribution of `train.csv`:

| Cause | Rows | Synthetic added |
|---|---:|---:|
| Equipment Failure | 551 | 0 |
| Corrosion Failure | 460 | 0 |
| Material Failure of Pipe or Weld | 300 | 40 |
| Excavation Damage | 300 | 47 |
| Natural Force Damage | 300 | 89 |
| Other Outside Force Damage | 300 | 153 |
| Other Incident Cause | 300 | 167 |
| Incorrect Operation | 300 | 185 |

The target is at least 300 training examples per class. Synthetic narratives
were created only for the six classes below that threshold, and only for the
training split; validation and test text was never used to create them.

## How the synthetic data was generated

`scripts/generate_synthetic.py` prompts a language model with 3 randomly chosen
real narratives of the target cause (from `data/experiment1/train.csv` only)
and asks for one new narrative in the same style. Each response is rejected
and regenerated unless it is lowercase, at least 50 words, not an exact copy of
a real or earlier synthetic narrative, and free of the cause name. Accepted
rows are saved one at a time to `data/processed/synthetic_narratives.csv`, so an
interrupted run resumes where it stopped. The script then builds `train.csv`
and checks: all 8 classes present, every class has at least 300 rows, no
validation or test narrative appears in train, all synthetic text is lowercase,
no synthetic row is an exact copy of a real one, and total rows equal real plus
synthetic. All checks passed.

**Generator models.** Free-tier daily limits meant no single model could write
all 681 rows, so several were used:

- `gemini-3.5-flash` (Gemini API): 16 rows, all Material Failure of Pipe or Weld.
- `openai/gpt-oss-120b`, `openai/gpt-oss-20b` and `qwen/qwen3.8-27b` (Groq API):
  the remaining 665 rows, split across classes. Per-row model attribution was
  not recorded, so it cannot be reported exactly.

To regenerate, put `GROQ_API_KEY` (or `GEMINI_API_KEY` with
`--provider gemini`) in the repository-root `.env` file, which Git ignores, and
run `python scripts/generate_synthetic.py [--groq-model MODEL]`. Do not put an
API key in source files or commit it.

The split and oversampling workflow uses random seed 42. Generated text is not
deterministic, so a fresh run will produce different narratives.
