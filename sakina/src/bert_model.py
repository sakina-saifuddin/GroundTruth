"""Frozen BERT encoder with a trainable multinomial classification head."""

from __future__ import annotations

import torch
from torch import nn
from transformers import BertModel


class FrozenBertClassifier(nn.Module):
    """Predict an incident cause from the mean-pooled token vectors of frozen BERT."""

    def __init__(
        self,
        num_labels: int = 8,
        model_name: str = "bert-base-uncased",
    ) -> None:
        """Load pretrained BERT, freeze its weights, and add a linear head."""
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        for parameter in self.bert.parameters():
            parameter.requires_grad = False

        hidden_size = int(self.bert.config.hidden_size)
        if hidden_size != 768:
            raise ValueError(
                f"{model_name} has hidden size {hidden_size}; this experiment expects BERT-base (768)."
            )
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch and classify each sequence using the attention-masked mean of its token vectors."""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        pooled = summed / mask.sum(dim=1).clamp(min=1e-9)
        return self.classifier(pooled)

    def train(self, mode: bool = True) -> "FrozenBertClassifier":
        """Set head mode while keeping frozen BERT dropout disabled."""
        super().train(mode)
        self.bert.eval()
        return self

    def count_trainable_params(self) -> int:
        """Count parameters that receive gradients during optimization."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
