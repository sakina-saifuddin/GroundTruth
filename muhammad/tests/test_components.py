"""Focused tests for Muhammad's BiLSTM-Attention components and result reporting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muhammad.src.bilstm_attention import AdditiveAttention, BiLSTMAttentionClassifier  # noqa: E402
from muhammad.src.comparison import generate_comparison  # noqa: E402
from muhammad.src.dataset import PHMSASequenceDataset, Vocabulary  # noqa: E402
from muhammad.src.evaluate import evaluate_model, save_results  # noqa: E402
from muhammad.src import train as train_module  # noqa: E402


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def test_vocabulary_pad_unk_always_at_zero_one():
    """PAD must be index 0 and UNK index 1 regardless of training text."""
    vocab = Vocabulary.build(["equipment failure corrosion", "pipe weld"])
    assert vocab.token2id[Vocabulary.PAD_TOKEN] == 0
    assert vocab.token2id[Vocabulary.UNK_TOKEN] == 1


def test_vocabulary_encode_pads_and_masks_correctly():
    """Short text is right-padded; mask is 1 for real tokens, 0 for pads."""
    vocab = Vocabulary.build(["hello world"])
    ids, mask = vocab.encode("hello world", max_length=5)
    assert len(ids) == 5
    assert len(mask) == 5
    assert mask == [1, 1, 0, 0, 0]
    assert ids[2] == Vocabulary.PAD_ID


def test_vocabulary_encode_truncates_long_text():
    """Text longer than max_length is silently truncated."""
    vocab = Vocabulary.build(["a b c d e f g"])
    ids, mask = vocab.encode("a b c d e f g", max_length=3)
    assert len(ids) == 3
    assert all(m == 1 for m in mask)


def test_vocabulary_save_and_load_roundtrip(tmp_path):
    """A vocabulary saved to disk and reloaded gives identical encodings."""
    vocab = Vocabulary.build(["corrosion failure pipe"])
    vocab.save(tmp_path / "vocab.json")
    vocab2 = Vocabulary.load(tmp_path / "vocab.json")
    assert vocab.token2id == vocab2.token2id
    ids1, mask1 = vocab.encode("corrosion pipe", max_length=4)
    ids2, mask2 = vocab2.encode("corrosion pipe", max_length=4)
    assert ids1 == ids2
    assert mask1 == mask2


def test_vocabulary_unknown_token_maps_to_unk():
    """Words not in the vocabulary map to UNK_ID, not KeyError."""
    vocab = Vocabulary.build(["known word"])
    ids, _ = vocab.encode("unknown_xyz_word", max_length=1)
    assert ids[0] == Vocabulary.UNK_ID


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def test_dataset_returns_three_long_tensors(tmp_path):
    """CSV examples become the expected fixed-length model inputs."""
    csv_path = tmp_path / "train.csv"
    pd.DataFrame({"narrative_clean": ["pipe failed leak"], "label": [3]}).to_csv(csv_path, index=False)
    vocab = Vocabulary.build(["pipe failed leak"])
    dataset = PHMSASequenceDataset(csv_path, vocab, max_length=10)

    item = dataset[0]
    assert set(item) == {"input_ids", "attention_mask", "label"}
    assert item["input_ids"].dtype == torch.long
    assert item["attention_mask"].dtype == torch.long
    assert item["input_ids"].shape == (10,)
    assert item["label"].item() == 3


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_bilstm_output_shape():
    """Model produces logits of (batch, num_labels) for a padded input batch."""
    model = BiLSTMAttentionClassifier(vocab_size=100, embed_dim=16, hidden_size=16, num_labels=8)
    input_ids = torch.randint(0, 100, (4, 20))
    mask = torch.ones(4, 20, dtype=torch.long)
    logits = model(input_ids=input_ids, attention_mask=mask)
    assert logits.shape == (4, 8)


def test_all_model_params_trainable():
    """BiLSTM-Attention has no frozen parameters — all are trainable by default."""
    model = BiLSTMAttentionClassifier(vocab_size=50, embed_dim=16, hidden_size=16, num_labels=8)
    assert all(p.requires_grad for p in model.parameters())
    assert model.count_trainable_params() > 0


def test_attention_masks_padding_positions():
    """AdditiveAttention assigns zero weight to padding positions."""
    attn = AdditiveAttention(hidden_size=8)
    hidden = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.long)
    _, weights = attn(hidden, mask)
    assert weights[0, 2].item() == pytest.approx(0.0, abs=1e-6)
    assert weights[1, 3].item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Evaluation and results
# ---------------------------------------------------------------------------

class PerfectTwoClassDataset(Dataset):
    """Small deterministic dataset where the model always predicts correctly."""

    def __len__(self):
        return 4

    def __getitem__(self, index):
        label = index % 2
        return {
            "input_ids": torch.tensor([label, 0], dtype=torch.long),
            "attention_mask": torch.tensor([1, 0], dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }


class PerfectTwoClassModel(nn.Module):
    def forward(self, input_ids, attention_mask):
        logits = torch.zeros(len(input_ids), 2)
        logits[torch.arange(len(input_ids)), input_ids[:, 0]] = 2.0
        return logits


def test_evaluate_and_save_metrics(tmp_path):
    """Evaluation produces exact scores and all three result files."""
    loader = DataLoader(PerfectTwoClassDataset(), batch_size=4)
    scores = evaluate_model(
        PerfectTwoClassModel(), loader, torch.device("cpu"), ["cause-a", "cause-b"]
    )
    assert scores["accuracy"] == 1.0
    assert scores["macro_f1"] == 1.0
    assert scores["micro_f1"] == 1.0
    assert np.array_equal(scores["confusion_matrix"], np.eye(2, dtype=int))

    save_results(scores, tmp_path, "test_exp")
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["accuracy"] == 1.0
    assert "cause-a" in (tmp_path / "per_class_report.txt").read_text(encoding="utf-8")
    assert (tmp_path / "confusion_matrix.png").is_file()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def test_comparison_does_not_invent_missing_results(tmp_path):
    """Comparison keeps unrun experiments visibly pending."""
    generate_comparison(tmp_path / "results", tmp_path / "comparison.md")
    comparison = (tmp_path / "comparison.md").read_text(encoding="utf-8")
    assert "Pending" in comparison
    assert "No experiment ranking" in comparison


# ---------------------------------------------------------------------------
# Training loop (monkeypatched with tiny helpers)
# ---------------------------------------------------------------------------

def test_train_model_saves_best_checkpoint_and_history(monkeypatch, tmp_path):
    """The training loop tracks validation scores and saves a loadable checkpoint."""

    # Tiny CSV that the real pd.read_csv inside train_model will read
    tiny_csv = tmp_path / "tiny.csv"
    pd.DataFrame({
        "narrative_clean": ["pipe failure corrosion", "excavation damage leak"],
        "label": [0, 1],
    }).to_csv(tiny_csv, index=False)

    class TinyVocab:
        token2id = {"<PAD>": 0, "<UNK>": 1, "word": 2}

        @classmethod
        def build(cls, texts, max_vocab_size=50_000, min_freq=1):
            return cls()

        @staticmethod
        def load(path):
            return TinyVocab()

        def save(self, path):
            Path(path).write_text("{}", encoding="utf-8")

        def __len__(self):
            return 3

    class TinyDataset(Dataset):
        def __init__(self, dataset_path, vocabulary, max_length):
            self.labels = [0, 1, 2, 3]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return {
                "input_ids": torch.tensor([index % 3], dtype=torch.long),
                "attention_mask": torch.ones(1, dtype=torch.long),
                "label": torch.tensor(self.labels[index] % 4, dtype=torch.long),
            }

    class TinyClassifier(nn.Module):
        def __init__(self, vocab_size, embed_dim, hidden_size, num_labels,
                     num_layers, dropout, padding_idx=0):
            super().__init__()
            self.linear = nn.Linear(1, num_labels)

        def count_trainable_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)

        def forward(self, input_ids, attention_mask):
            return self.linear(input_ids.float())

    monkeypatch.setattr(train_module, "Vocabulary", TinyVocab)
    monkeypatch.setattr(train_module, "PHMSASequenceDataset", TinyDataset)
    monkeypatch.setattr(train_module, "BiLSTMAttentionClassifier", TinyClassifier)

    result = train_module.train_model(
        {
            "train_csv": str(tiny_csv),
            "val_csv": str(tiny_csv),
            "experiment_name": "unit_test",
            "checkpoint_dir": str(tmp_path),
            "max_length": 5,
            "max_vocab_size": 10,
            "embed_dim": 8,
            "hidden_size": 8,
            "num_layers": 1,
            "dropout": 0.0,
            "batch_size": 2,
            "epochs": 2,
            "patience": 2,
            "use_class_weights": False,
            "seed": 42,
            "num_labels": 4,
        }
    )

    assert result["best_epoch"] in (1, 2)
    assert len(result["history"]) == 2
    assert result["trainable_parameters"] > 0
    checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
    assert checkpoint["best_epoch"] == result["best_epoch"]
