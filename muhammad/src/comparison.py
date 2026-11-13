"""Generate a factual comparison from completed BiLSTM-Attention experiment metric files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EXPERIMENTS = (
    ("A", "Experiment A (class weights)", "experiment_a"),
    ("B", "Experiment B (oversampling)", "experiment_b"),
)
CAUSE_LABELS = [
    "EQUIPMENT FAILURE",
    "CORROSION FAILURE",
    "MATERIAL FAILURE OF PIPE OR WELD",
    "EXCAVATION DAMAGE",
    "NATURAL FORCE DAMAGE",
    "OTHER OUTSIDE FORCE DAMAGE",
    "OTHER INCIDENT CAUSE",
    "INCORRECT OPERATION",
]


def _load_metrics(path: Path) -> dict[str, Any] | None:
    """Load an experiment's metrics, returning None until the run exists."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def generate_comparison(results_root: str | Path, output_path: str | Path) -> None:
    """Write side-by-side metrics and observations without inventing missing results."""
    root = Path(results_root)
    output = Path(output_path)
    metrics = {
        key: _load_metrics(root / folder / "metrics.json")
        for key, _, folder in EXPERIMENTS
    }
    labels = [
        label
        for label in CAUSE_LABELS
        if any(result and label in result.get("per_class", {}) for result in metrics.values())
    ]
    if not labels:
        labels = CAUSE_LABELS

    lines = [
        "# Experiment comparison",
        "",
        "This file is generated from the saved test metrics after each experiment. "
        "Missing runs are shown as pending, not estimated.",
        "",
        "| Metric | Experiment A (class weights) | Experiment B (oversampling) |",
        "|---|---:|---:|",
    ]
    metric_rows = [
        ("Test accuracy", "accuracy"),
        ("Test macro-F1", "macro_f1"),
        ("Test micro-F1", "micro_f1"),
        ("Best val macro-F1", "best_val_f1"),
        ("Best epoch", "best_epoch"),
        ("Trainable parameters", "trainable_parameters"),
    ]
    for title, key in metric_rows:
        cells = []
        for experiment in ("A", "B"):
            result = metrics[experiment]
            value = None if result is None else result.get(key)
            if value is None:
                cells.append("Pending")
            elif key in ("best_epoch", "trainable_parameters"):
                cells.append(str(value))
            else:
                cells.append(f"{value:.4f}")
        lines.append(f"| {title} | {cells[0]} | {cells[1]} |")

    lines.extend(
        [
            "",
            "## Per-class test F1",
            "",
            "| Cause | Experiment A | Experiment B |",
            "|---|---:|---:|",
        ]
    )
    for label in labels:
        cells = []
        for experiment in ("A", "B"):
            result = metrics[experiment]
            scores = None if result is None else result.get("per_class", {}).get(label)
            cells.append("Pending" if scores is None else f"{scores['f1']:.4f}")
        lines.append(f"| {label} | {cells[0]} | {cells[1]} |")

    if all(metrics.values()):
        score_a = float(metrics["A"]["macro_f1"])
        score_b = float(metrics["B"]["macro_f1"])
        if score_a == score_b:
            overall = "Both experiments have the same test macro-F1."
        else:
            better = "Experiment A (class weights)" if score_a > score_b else "Experiment B (oversampling)"
            overall = (
                f"{better} has the higher test macro-F1 "
                f"({max(score_a, score_b):.4f} vs. {min(score_a, score_b):.4f})."
            )

        differences = []
        for label in labels:
            f1_a = float(metrics["A"]["per_class"][label]["f1"])
            f1_b = float(metrics["B"]["per_class"][label]["f1"])
            differences.append((label, f1_b - f1_a))
        helped = sorted((item for item in differences if item[1] > 0), key=lambda item: item[1], reverse=True)
        lower = sorted((item for item in differences if item[1] < 0), key=lambda item: item[1])
        improved_text = (
            ", ".join(f"{label} (+{change:.4f})" for label, change in helped[:3])
            if helped
            else "none; no class had higher test F1 with oversampling"
        )
        decreased_text = (
            ", ".join(f"{label} ({change:.4f})" for label, change in lower[:3])
            if lower
            else "none; no class had lower test F1 with oversampling"
        )
        paragraph = (
            f"{overall} Compared with class weighting, oversampling had higher test F1 for: "
            f"{improved_text}. It had lower test F1 for: {decreased_text}. These are measured "
            "test-set differences, not evidence that oversampling caused the change."
        )
    else:
        missing = [name for key, name, _ in EXPERIMENTS if metrics[key] is None]
        paragraph = (
            f"Comparison is pending because metrics are not available for {', '.join(missing)}. "
            "No experiment ranking or class-benefit claims are made until both runs finish."
        )
    lines.extend(["", "## Summary", "", paragraph, ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
