"""Training loop for the BiLSTM-Attention classifier."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from .bilstm_attention import BiLSTMAttentionClassifier
from .dataset import PHMSASequenceDataset, Vocabulary


def _load_fasttext_embeddings(
    fasttext_path: str | Path,
    vocabulary: Vocabulary,
    embed_dim: int,
) -> np.ndarray:
    """Build an embedding matrix from a FastText .bin model for the given vocabulary.

    Uses the fasttext library so every word — including OOV technical terms —
    gets a vector computed from its character n-grams. PAD (index 0) is zeroed out.
    """
    path = Path(fasttext_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"FastText model not found: {path}\n"
            "Run:  python muhammad/scripts/download_fasttext.py"
        )
    try:
        import fasttext
    except ImportError as exc:
        raise ImportError("fasttext is not installed. Run:  pip install fasttext") from exc

    print(f"Loading FastText model from {path} (this may take ~30 seconds) ...")
    ft = fasttext.load_model(str(path))

    matrix = np.zeros((len(vocabulary), embed_dim), dtype=np.float32)
    for token, idx in vocabulary.token2id.items():
        if idx == Vocabulary.PAD_ID:
            continue
        vec = ft.get_word_vector(token)
        if len(vec) != embed_dim:
            raise ValueError(
                f"FastText vector dim {len(vec)} does not match embed_dim {embed_dim}."
            )
        matrix[idx] = vec

    print(
        f"FastText: built embedding matrix for {len(vocabulary)} vocabulary words "
        f"(all words covered via subword n-grams)."
    )
    return matrix


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and available PyTorch devices for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _validation_metrics(
    model: BiLSTMAttentionClassifier,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Return mean validation loss and macro-F1 for one epoch."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    labels: list[int] = []
    predictions: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            batch_labels = batch["label"].to(device)
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            batch_loss = loss_function(logits, batch_labels)
            total_loss += float(batch_loss.item()) * len(batch_labels)
            total_examples += len(batch_labels)
            labels.extend(batch_labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    if not total_examples:
        raise ValueError("Validation split is empty.")
    macro_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
    return total_loss / total_examples, macro_f1


def train_model(config: dict[str, Any]) -> dict[str, Any]:
    """Build vocabulary, train the BiLSTM-Attention head, and save the best checkpoint.

    Args:
        config: Experiment configuration dictionary. Expected keys:
            train_csv, val_csv, experiment_name, checkpoint_dir,
            max_length, max_vocab_size, embed_dim, hidden_size,
            num_layers, dropout, batch_size, lr, epochs, patience,
            use_class_weights, class_weights (path), seed, num_labels.

    Returns:
        dict with best_val_f1, best_epoch, history, checkpoint_path,
        vocab_path, and trainable_parameters.
    """
    seed = int(config.get("seed", 42))
    set_seed(seed)

    max_length = int(config.get("max_length", 300))
    max_vocab_size = int(config.get("max_vocab_size", 50_000))
    embed_dim = int(config.get("embed_dim", 128))
    hidden_size = int(config.get("hidden_size", 128))
    num_layers = int(config.get("num_layers", 2))
    dropout = float(config.get("dropout", 0.3))
    batch_size = int(config.get("batch_size", 32))
    epochs = int(config.get("epochs", 30))
    patience = int(config.get("patience", 5))
    learning_rate = float(config.get("lr", 1e-3))
    num_labels = int(config.get("num_labels", 8))

    if batch_size < 1 or epochs < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("batch_size, epochs, patience, and lr must all be positive.")

    import pandas as pd

    train_df = pd.read_csv(config["train_csv"])
    if train_df["narrative_clean"].isna().any():
        raise ValueError("Training CSV contains empty narratives.")
    vocabulary = Vocabulary.build(
        train_df["narrative_clean"].tolist(),
        max_vocab_size=max_vocab_size,
    )

    checkpoint_dir = Path(config.get("checkpoint_dir", "muhammad/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = str(config.get("experiment_name", "experiment"))
    vocab_path = checkpoint_dir / f"vocab_{experiment_name}.json"
    vocabulary.save(vocab_path)

    train_dataset = PHMSASequenceDataset(config["train_csv"], vocabulary, max_length)
    val_dataset = PHMSASequenceDataset(config["val_csv"], vocabulary, max_length)
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fasttext_path = config.get("fasttext_path")
    freeze_embeddings = bool(config.get("freeze_embeddings", fasttext_path is not None))
    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocabulary),
        embed_dim=embed_dim,
        hidden_size=hidden_size,
        num_labels=num_labels,
        num_layers=num_layers,
        dropout=dropout,
        freeze_embeddings=freeze_embeddings,
    )
    if fasttext_path:
        embedding_matrix = _load_fasttext_embeddings(fasttext_path, vocabulary, embed_dim)
        model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        model.embedding.weight.data[Vocabulary.PAD_ID].zero_()
    trainable_parameters = model.count_trainable_params()
    model.to(device)

    weight_tensor = None
    if config.get("use_class_weights", False):
        weights_path = Path(config["class_weights"])
        if not weights_path.is_file():
            raise FileNotFoundError(f"Training class weights were not found: {weights_path}")
        weights_data = json.loads(weights_path.read_text(encoding="utf-8"))
        ordered_weights = weights_data.get("ordered_by_label")
        if ordered_weights is None:
            raise ValueError(f"{weights_path} does not contain ordered_by_label.")
        if len(ordered_weights) != num_labels:
            raise ValueError("The number of class weights does not match num_labels.")
        weight_tensor = torch.tensor(ordered_weights, dtype=torch.float32, device=device)

    loss_function = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    checkpoint_path = checkpoint_dir / f"best_{experiment_name}.pt"
    best_val_f1 = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_total = 0.0
        train_examples = 0
        for batch in train_loader:
            batch_labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = loss_function(logits, batch_labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_total += float(loss.item()) * len(batch_labels)
            train_examples += len(batch_labels)

        if not train_examples:
            raise ValueError("Training split is empty.")
        mean_train_loss = train_loss_total / train_examples
        val_loss, val_macro_f1 = _validation_metrics(model, val_loader, loss_function, device)
        epoch_metrics: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": mean_train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
        }
        history.append(epoch_metrics)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss: {mean_train_loss:.4f} | "
            f"val loss: {val_loss:.4f} | "
            f"val macro-F1: {val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                    "config": config,
                    "vocab_size": len(vocabulary),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"Early stopping after {patience} epochs without "
                    "validation macro-F1 improvement."
                )
                break

    if best_epoch == 0:
        raise RuntimeError("Training ended without saving a best checkpoint.")
    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "vocab_path": str(vocab_path),
        "trainable_parameters": trainable_parameters,
    }
