# GroundTruth
Because every good label needs solid ground. Classifying gas pipeline incident causes with BiGRU, BiLSTM and BERT.

## Shared data preparation

One record with an empty narrative was removed, leaving 3,026 narratives.

### The planned split

The split sizes below are produced by `python scripts/prepare_data.py`.

| Split | Share | Rows |
|---|---:|---:|
| Train | 70% | 2,118 |
| Validation | 15% | 453 |
| Test | 15% | 455 |

### Reproduce the data split

```powershell
pip install -r requirements.txt
python scripts/prepare_data.py
```
