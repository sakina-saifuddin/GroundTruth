"""Vocabulary builder and dataset wrapper for PHMSA narrative CSV files."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset


_WHITESPACE_RE = re.compile(r"\s+")


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on whitespace."""
    return _WHITESPACE_RE.split(text.lower().strip())


class Vocabulary:
    """Map words to integer IDs with special PAD and UNK tokens."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    PAD_ID = 0
    UNK_ID = 1

    def __init__(self) -> None:
        self.token2id: dict[str, int] = {}
        self.id2token: dict[int, str] = {}
        self._next_id = 0

    def _add(self, token: str) -> None:
        if token not in self.token2id:
            self.token2id[token] = self._next_id
            self.id2token[self._next_id] = token
            self._next_id += 1

    @classmethod
    def build(
        cls,
        texts: list[str],
        max_vocab_size: int = 50_000,
        min_freq: int = 1,
    ) -> "Vocabulary":
        """Build a vocabulary from a list of raw text strings.

        PAD and UNK are always added first at indices 0 and 1.
        Words are ranked by frequency; ties are broken by first occurrence.
        """
        vocab = cls()
        vocab._add(cls.PAD_TOKEN)
        vocab._add(cls.UNK_TOKEN)

        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(_tokenize(text))

        for token, count in counts.most_common():
            if count < min_freq:
                break
            if len(vocab) >= max_vocab_size:
                break
            vocab._add(token)
        return vocab

    def encode(self, text: str, max_length: int) -> tuple[list[int], list[int]]:
        """Convert text to a fixed-length integer sequence and an attention mask.

        Returns:
            ids: list of length max_length, padded with PAD_ID.
            mask: list of length max_length, 1 for real tokens, 0 for padding.
        """
        tokens = _tokenize(text)[:max_length]
        ids = [self.token2id.get(t, self.UNK_ID) for t in tokens]
        pad_len = max_length - len(ids)
        mask = [1] * len(ids) + [0] * pad_len
        ids = ids + [self.PAD_ID] * pad_len
        return ids, mask

    def __len__(self) -> int:
        return len(self.token2id)

    def save(self, path: str | Path) -> None:
        """Persist the vocabulary to a JSON file."""
        Path(path).write_text(
            json.dumps({"token2id": self.token2id}, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Restore a vocabulary from a JSON file saved by :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab = cls()
        vocab.token2id = data["token2id"]
        vocab.id2token = {v: k for k, v in vocab.token2id.items()}
        vocab._next_id = max(vocab.token2id.values()) + 1 if vocab.token2id else 0
        return vocab


class PHMSASequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Word-sequence dataset for BiLSTM experiments."""

    def __init__(
        self,
        dataset_path: str | Path,
        vocabulary: Vocabulary,
        max_length: int = 300,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"Dataset CSV was not found: {self.dataset_path}")

        self.data = pd.read_csv(self.dataset_path)
        required = {"narrative_clean", "label"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(
                f"{self.dataset_path} is missing required columns: {sorted(missing)}"
            )
        if self.data["narrative_clean"].isna().any():
            raise ValueError(f"{self.dataset_path} contains an empty cleaned narrative.")
        if self.data["label"].isna().any():
            raise ValueError(f"{self.dataset_path} contains a missing label.")
        if max_length < 1:
            raise ValueError("max_length must be at least 1.")

        self.data = self.data.reset_index(drop=True)
        self.vocabulary = vocabulary
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return tokenized input IDs, attention mask, and integer label."""
        row = self.data.iloc[index]
        ids, mask = self.vocabulary.encode(str(row["narrative_clean"]), self.max_length)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }
