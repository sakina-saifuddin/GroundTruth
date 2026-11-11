"""Dataset wrapper for the prepared PHMSA narrative CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class PHMSADataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenize cleaned narratives from a Phase 1 CSV for frozen BERT."""

    def __init__(
        self,
        dataset_path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
    ) -> None:
        """Read a prepared split and retain its tokenizer and sequence limit."""
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"Dataset CSV was not found: {self.dataset_path}")

        self.data = pd.read_csv(self.dataset_path)
        required_columns = {"narrative_clean", "label"}
        missing_columns = required_columns.difference(self.data.columns)
        if missing_columns:
            raise ValueError(
                f"{self.dataset_path} is missing required columns: {sorted(missing_columns)}"
            )
        if self.data["narrative_clean"].isna().any():
            raise ValueError(f"{self.dataset_path} contains an empty cleaned narrative.")
        if self.data["label"].isna().any():
            raise ValueError(f"{self.dataset_path} contains a missing label.")
        if max_length < 2:
            raise ValueError("max_length must leave room for BERT special tokens.")

        self.data = self.data.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of examples in this split."""
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Tokenize one narrative and return model inputs and its integer label."""
        row = self.data.iloc[index]
        encoded: dict[str, Any] = self.tokenizer(
            str(row["narrative_clean"]),
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        item = {
            key: encoded[key].squeeze(0).to(dtype=torch.long)
            for key in ("input_ids", "attention_mask", "token_type_ids")
        }
        item["label"] = torch.tensor(int(row["label"]), dtype=torch.long)
        return item
