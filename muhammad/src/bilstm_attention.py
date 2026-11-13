"""BiLSTM encoder with additive attention and a multinomial classification head."""

from __future__ import annotations

import torch
from torch import nn


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention over BiLSTM hidden states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.query = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the attention-weighted context vector and the attention weights.

        Args:
            hidden_states: (batch, seq_len, hidden_size) BiLSTM outputs.
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding.

        Returns:
            context: (batch, hidden_size) weighted sum of hidden states.
            weights: (batch, seq_len) attention distribution.
        """
        energy = torch.tanh(self.projection(hidden_states))
        scores = self.query(energy).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        context = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return context, weights


class BiLSTMAttentionClassifier(nn.Module):
    """Predict an incident cause from a word-level BiLSTM with additive attention.

    Architecture:
        1. Trainable word embeddings
        2. Bidirectional LSTM (num_layers stacked)
        3. Additive attention pooling over all time steps
        4. Dropout + linear classification head
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 300,
        hidden_size: int = 128,
        num_labels: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
        freeze_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden_size=2 * hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(2 * hidden_size, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch and classify via attention-pooled BiLSTM representation.

        Args:
            input_ids: (batch, seq_len) integer token IDs.
            attention_mask: (batch, seq_len) — 1 for real tokens, 0 for padding.

        Returns:
            Logits of shape (batch, num_labels).
        """
        embedded = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(embedded)
        context, _ = self.attention(lstm_out, attention_mask)
        return self.classifier(self.dropout(context))

    def count_trainable_params(self) -> int:
        """Count parameters that receive gradients during optimization."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
