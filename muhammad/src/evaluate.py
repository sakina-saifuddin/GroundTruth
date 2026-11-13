"""Evaluation and result-file utilities for Muhammad's BiLSTM-Attention experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    label_names: list[str],
) -> dict[str, Any]:
    """Calculate overall and per-class scores plus the raw confusion matrix."""
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)
            logits = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())

    if not all_labels:
        raise ValueError("Cannot evaluate an empty dataset.")
    label_ids = list(range(len(label_names)))
    matrix = confusion_matrix(all_labels, all_predictions, labels=label_ids)
    class_report = classification_report(
        all_labels,
        all_predictions,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(class_report[name]["precision"]),
            "recall": float(class_report[name]["recall"]),
            "f1": float(class_report[name]["f1-score"]),
            "support": int(class_report[name]["support"]),
        }
        for name in label_names
    }

    return {
        "accuracy": float(accuracy_score(all_labels, all_predictions)),
        "macro_f1": float(
            f1_score(all_labels, all_predictions, labels=label_ids, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(all_labels, all_predictions, labels=label_ids, average="micro", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def save_results(
    results: dict[str, Any],
    out_dir: str | Path,
    experiment_name: str,
) -> None:
    """Save metrics JSON, a readable report, and a row-normalized confusion matrix heatmap."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.asarray(results["confusion_matrix"], dtype=np.int64)
    labels = list(results["per_class"])
    metrics_for_json = {
        key: value for key, value in results.items() if key != "confusion_matrix"
    }
    metrics_for_json["confusion_matrix"] = matrix.tolist()
    metrics_for_json["experiment_name"] = experiment_name
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_for_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    report_lines = [
        f"Evaluation report: {experiment_name}",
        "",
        f"Accuracy: {results['accuracy']:.6f}",
        f"Macro-F1: {results['macro_f1']:.6f}",
        f"Micro-F1: {results['micro_f1']:.6f}",
        "",
        f"{'Cause':40} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}",
        "-" * 85,
    ]
    for label, scores in results["per_class"].items():
        report_lines.append(
            f"{label:40} {scores['precision']:10.4f} {scores['recall']:10.4f} "
            f"{scores['f1']:10.4f} {scores['support']:10d}"
        )
    (output_dir / "per_class_report.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    row_totals = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix.astype(float),
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    figure_width = max(10, len(labels) * 1.15)
    figure, axis = plt.subplots(figsize=(figure_width, figure_width * 0.8))
    sns.heatmap(
        normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        vmin=0,
        vmax=1,
        ax=axis,
    )
    axis.set_title(f"{experiment_name}: confusion matrix normalized by true cause")
    axis.set_xlabel("Predicted cause")
    axis.set_ylabel("True cause")
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)
