# Shared data files

The original PHMSA workbook belongs in `data/raw/`. It is the only input to
Phase 1; the `Summary` sheet is metadata and is not used for modeling. Run
`python scripts/prepare_data.py` from the repository root to regenerate the
processed files. The split uses a fixed random seed of `42`.

## Files

| File | Contents |
|---|---|
| `raw/PHMSA_Gas_Transmission_Gathering_Merged_2002_Present.xlsx` | Public source workbook; do not edit the source copy. |
| `processed/clean_full.csv` | All rows with a non-empty narrative, original source columns, cleaned narrative, and integer label. |
| `processed/train.csv` | Training rows and model labels. |
| `processed/val.csv` | Validation rows and model labels. |
| `processed/test.csv` | Held-out test rows and model labels. |
| `processed/label_map.json` | Cause-to-id and id-to-cause mappings. |
| `processed/class_weights_train.json` | Class weights computed from the training split, both by cause and in label order. |
| `processed/split_manifest.csv` | `REPORT_NUMBER` to split assignment for exact split reproduction. |
| `../reports/data_report.md` | Generated row counts, cleaning, distribution, class weights, word lengths, and optional BERT token statistics. |
| `../reports/figures/` | Generated EDA charts. |

Split membership is assigned by cleaned narrative, not by individual row.
Duplicate narratives therefore stay together to avoid information leakage.
Synthetic narratives, if created in a later experiment, may only be created
from `train.csv`; never create synthetic examples from validation or test.

## Loading the files

```python
import json
import pandas as pd
import torch

train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

with open("data/processed/label_map.json", encoding="utf-8") as file:
    labels = json.load(file)

with open("data/processed/class_weights_train.json", encoding="utf-8") as file:
    class_weight_data = json.load(file)

weights = torch.tensor(class_weight_data["ordered_by_label"], dtype=torch.float32)
loss_function = torch.nn.CrossEntropyLoss(weight=weights)
```

The model input is `narrative_clean`; `label` is the integer target. `NAME`,
`CAUSE_DETAILS`, and `REPORT_NUMBER` are retained for auditability and should
not be passed as model features. Install PyTorch separately if you want to run
the example loss function; it is not needed for data preparation.
