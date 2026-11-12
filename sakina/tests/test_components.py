"""Focused tests for the Sakina classifier components and result reporting."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sakina.scripts.oversample import build_augmented_train  # noqa: E402
from sakina.src.bert_dataset import PHMSADataset  # noqa: E402
from sakina.src.bert_model import FrozenBertClassifier  # noqa: E402
from sakina.src.comparison import generate_comparison  # noqa: E402
from sakina.src.evaluate import evaluate_model, save_results  # noqa: E402
from sakina.src import train as train_module  # noqa: E402


class TinyTokenizer:
    """Return small tensor-like token batches for dataset unit tests."""

    def __call__(self, text: str, **kwargs):
        length = kwargs["max_length"]
        return {
            "input_ids": torch.ones((1, length), dtype=torch.long),
            "attention_mask": torch.ones((1, length), dtype=torch.long),
            "token_type_ids": torch.zeros((1, length), dtype=torch.long),
        }


def test_dataset_returns_four_long_tensors(tmp_path):
    """CSV examples become the expected fixed-length model inputs."""
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"narrative_clean": ["pipe failed"], "label": [3]}).to_csv(csv_path, index=False)
    dataset = PHMSADataset(csv_path, TinyTokenizer(), max_length=12)

    item = dataset[0]
    assert set(item) == {"input_ids", "attention_mask", "token_type_ids", "label"}
    assert all(value.dtype == torch.long for value in item.values())
    assert item["input_ids"].shape == (12,)
    assert item["label"].item() == 3


def test_model_freezes_bert_and_trains_exact_head():
    """Frozen BERT contributes no trainable weights; the eight-class head has 6,152."""

    class FakeBert(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder_weight = nn.Parameter(torch.ones(4))
            self.config = SimpleNamespace(hidden_size=768)

        def forward(self, input_ids, **kwargs):
            batch_size = input_ids.shape[0]
            return SimpleNamespace(pooler_output=torch.ones(batch_size, 768))

    with patch("sakina.src.bert_model.BertModel.from_pretrained", return_value=FakeBert()):
        model = FrozenBertClassifier()

    logits = model(
        input_ids=torch.ones((2, 4), dtype=torch.long),
        attention_mask=torch.ones((2, 4), dtype=torch.long),
        token_type_ids=torch.zeros((2, 4), dtype=torch.long),
    )
    assert logits.shape == (2, 8)
    assert model.count_trainable_params() == 6152
    assert all(not parameter.requires_grad for parameter in model.bert.parameters())
    model.train()
    assert not model.bert.training


class TwoClassDataset(Dataset):
    """Small deterministic dataset used to check metric serialization."""

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([index], dtype=torch.long),
            "attention_mask": torch.ones(1, dtype=torch.long),
            "token_type_ids": torch.zeros(1, dtype=torch.long),
            "label": torch.tensor(index, dtype=torch.long),
        }


class TwoClassModel(nn.Module):
    """Return a known correct prediction for each test item."""

    def forward(self, input_ids, **kwargs):
        logits = torch.zeros((len(input_ids), 2))
        logits[torch.arange(len(input_ids)), input_ids[:, 0]] = 2
        return logits


def test_evaluate_and_save_metrics(tmp_path):
    """Evaluation produces exact scores and all three result files."""
    loader = DataLoader(TwoClassDataset(), batch_size=2)
    scores = evaluate_model(TwoClassModel(), loader, torch.device("cpu"), ["cause-a", "cause-b"])
    assert scores["accuracy"] == 1.0
    assert scores["macro_f1"] == 1.0
    assert scores["micro_f1"] == 1.0
    assert np.array_equal(scores["confusion_matrix"], np.eye(2, dtype=int))

    save_results(scores, tmp_path, "test")
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["accuracy"] == 1.0
    assert "cause-a" in (tmp_path / "per_class_report.txt").read_text(encoding="utf-8")
    assert (tmp_path / "confusion_matrix.png").is_file()


def test_comparison_does_not_invent_missing_results(tmp_path):
    """Comparison keeps unrun experiments visibly pending."""
    generate_comparison(tmp_path / "results", tmp_path / "comparison.md")
    comparison = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "Pending" in comparison
    assert "No experiment ranking" in comparison


def test_oversampling_fails_clearly_without_synthetic_file(tmp_path):
    """Experiment B never silently creates synthetic narratives."""
    train_csv = tmp_path / "train.csv"
    train_csv.write_text("narrative_clean,CAUSE,label\ntext,EQUIPMENT FAILURE,0\n", encoding="utf-8")
    label_map = tmp_path / "label_map.json"
    label_map.write_text(
        json.dumps({"label2id": {"EQUIPMENT FAILURE": 0}}),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="Synthetic training file is required"):
        build_augmented_train(train_csv, label_map, tmp_path / "augmented.csv", tmp_path / "missing.csv")


def test_oversampling_fills_all_rare_classes_without_leakage(tmp_path):
    """Supplied synthetic rows fill every below-300 class and preserve train columns."""
    train_path = ROOT / "data" / "processed" / "train.csv"
    label_map_path = ROOT / "data" / "processed" / "label_map.json"
    train = pd.read_csv(train_path)
    labels = json.loads(label_map_path.read_text(encoding="utf-8"))["label2id"]
    counts = train["CAUSE"].value_counts()
    synthetic_rows = []
    for cause, label in labels.items():
        missing = max(0, 300 - int(counts.get(cause, 0)))
        for index in range(missing):
            synthetic_rows.append(
                {
                    "narrative_clean": f"unit test synthetic narrative {cause.lower()} number {index}",
                    "CAUSE": cause,
                    "label": label,
                }
            )
    synthetic_path = tmp_path / "synthetic.csv"
    pd.DataFrame(synthetic_rows).to_csv(synthetic_path, index=False)
    output_path = tmp_path / "augmented.csv"

    augmented = build_augmented_train(
        train_path,
        label_map_path,
        output_path,
        synthetic_path,
    )
    assert output_path.is_file()
    assert set(augmented.columns) == set(train.columns)
    for cause, count in augmented["CAUSE"].value_counts().items():
        if int(train["CAUSE"].value_counts().get(cause, 0)) < 300:
            assert count == 300
    synthetic_text = set(pd.read_csv(synthetic_path)["narrative_clean"])
    assert synthetic_text.isdisjoint(set(pd.read_csv(ROOT / "data" / "processed" / "val.csv")["narrative_clean"]))
    assert synthetic_text.isdisjoint(set(pd.read_csv(ROOT / "data" / "processed" / "test.csv")["narrative_clean"]))
    assert augmented["NARRATIVE"].notna().all()


def test_train_model_saves_best_checkpoint_and_history(monkeypatch, tmp_path):
    """The training loop tracks validation scores and saves a loadable checkpoint."""

    class TinyDataset(Dataset):
        def __init__(self, dataset_path, tokenizer, max_length):
            self.labels = [0, 1, 2, 3]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {
                "input_ids": torch.tensor([float(index)], dtype=torch.float32),
                "attention_mask": torch.ones(1, dtype=torch.long),
                "token_type_ids": torch.zeros(1, dtype=torch.long),
                "label": torch.tensor(self.labels[index], dtype=torch.long),
            }

    class TinyClassifier(nn.Module):
        def __init__(self, num_labels, model_name):
            super().__init__()
            self.linear = nn.Linear(1, num_labels)

        def count_trainable_params(self):
            return 768 * 8 + 8

        def forward(self, input_ids, **kwargs):
            return self.linear(input_ids.float())

    monkeypatch.setattr(train_module.AutoTokenizer, "from_pretrained", lambda model_name: object())
    monkeypatch.setattr(train_module, "PHMSADataset", TinyDataset)
    monkeypatch.setattr(train_module, "FrozenBertClassifier", TinyClassifier)
    result = train_module.train_model(
        {
            "train_csv": "not-read-by-test.csv",
            "val_csv": "not-read-by-test.csv",
            "experiment_name": "unit_test",
            "checkpoint_dir": str(tmp_path),
            "batch_size": 2,
            "epochs": 2,
            "patience": 2,
            "use_class_weights": False,
            "seed": 42,
        }
    )

    assert result["trainable_parameters"] == 6152
    assert result["best_epoch"] in (1, 2)
    assert len(result["history"]) == 2
    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
    assert checkpoint["best_epoch"] == result["best_epoch"]
