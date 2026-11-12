"""Training loop for frozen BERT and its trainable classification head."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .bert_dataset import PHMSADataset
from .bert_model import FrozenBertClassifier


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
    model: FrozenBertClassifier,
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
                token_type_ids=batch["token_type_ids"].to(device),
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
    """Train the linear head, early-stop on validation macro-F1, and save its best weights."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    model_name = str(config.get("model_name", "bert-base-uncased"))
    max_length = int(config.get("max_length", 512))
    batch_size = int(config.get("batch_size", 16))
    epochs = int(config.get("epochs", 20))
    patience = int(config.get("patience", 5))
    learning_rate = float(config.get("lr", 2e-3))
    if batch_size < 1 or epochs < 1 or patience < 1 or learning_rate <= 0:
        raise ValueError("batch_size, epochs, patience, and lr must all be positive.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = PHMSADataset(config["train_csv"], tokenizer, max_length=max_length)
    val_dataset = PHMSADataset(config["val_csv"], tokenizer, max_length=max_length)
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
    model = FrozenBertClassifier(
        num_labels=int(config.get("num_labels", 8)),
        model_name=model_name,
        pooling=str(config.get("pooling", "mean")),
    )
    trainable_parameters = model.count_trainable_params()
    expected_parameters = 768 * int(config.get("num_labels", 8)) + int(config.get("num_labels", 8))
    if trainable_parameters != expected_parameters:
        raise RuntimeError(
            f"Expected {expected_parameters:,} trainable head parameters, got {trainable_parameters:,}."
        )
    if int(config.get("num_labels", 8)) == 8 and trainable_parameters != 6152:
        raise RuntimeError(f"BERT-base classification head must have 6,152 trainable parameters, got {trainable_parameters}.")
    model.to(device)

    weight_tensor = None
    if config.get("use_class_weights", False):
        weights_path = Path(config["class_weights"])
        if not weights_path.is_file():
            raise FileNotFoundError(f"Training class weights were not found: {weights_path}")
        import json

        weights_data = json.loads(weights_path.read_text(encoding="utf-8"))
        ordered_weights = weights_data.get("ordered_by_label")
        if ordered_weights is None:
            raise ValueError(f"{weights_path} does not contain ordered_by_label.")
        if len(ordered_weights) != int(config.get("num_labels", 8)):
            raise ValueError("The number of class weights does not match num_labels.")
        weight_tensor = torch.tensor(ordered_weights, dtype=torch.float32, device=device)

    loss_function = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )

    checkpoint_dir = Path(config.get("checkpoint_dir", "sakina/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = str(config.get("experiment_name", "experiment"))
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
            labels = batch["label"].to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                token_type_ids=batch["token_type_ids"].to(device),
            )
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item()) * len(labels)
            train_examples += len(labels)

        if not train_examples:
            raise ValueError("Training split is empty.")
        mean_train_loss = train_loss_total / train_examples
        val_loss, val_macro_f1 = _validation_metrics(model, val_loader, loss_function, device)
        epoch_metrics = {
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
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without validation macro-F1 improvement.")
                break

    if best_epoch == 0:
        raise RuntimeError("Training ended without saving a best checkpoint.")
    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
        "trainable_parameters": trainable_parameters,
    }
