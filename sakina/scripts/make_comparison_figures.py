"""Build comparison figures for the four frozen-BERT runs from saved metrics.json files."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "reports" / "figures"
RUNS = {
    "V1 pooler / A (class weights)": ROOT / "sakina" / "results" / "experiment_a" / "metrics.json",
    "V1 pooler / B (oversampling)": ROOT / "sakina" / "results" / "experiment_b" / "metrics.json",
    "V2 mean pool / A (class weights)": ROOT / "sakina" / "results_mean_pooling" / "experiment_a" / "metrics.json",
    "V2 mean pool / B (oversampling)": ROOT / "sakina" / "results_mean_pooling" / "experiment_b" / "metrics.json",
}
COLORS = ["#9ecae1", "#fdae6b", "#3182bd", "#e6550d"]


def _load() -> dict[str, dict]:
    return {name: json.loads(path.read_text(encoding="utf-8")) for name, path in RUNS.items()}


def overall_figure(data: dict[str, dict]) -> None:
    """Bar chart of test accuracy and macro-F1 for all four runs."""
    names = list(data)
    x = np.arange(2)
    width = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(names):
        values = [data[name]["accuracy"], data[name]["macro_f1"]]
        bars = ax.bar(x + (i - 1.5) * width, values, width, label=name, color=COLORS[i])
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, ["Test accuracy", "Test macro-F1"])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Score")
    ax.set_title("Frozen BERT + linear head: pooling method x imbalance method")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overall_comparison.png", dpi=160)
    plt.close(fig)


def per_class_figure(data: dict[str, dict]) -> None:
    """Grouped bars of per-class test F1 for all four runs."""
    names = list(data)
    classes = list(next(iter(data.values()))["per_class"])
    x = np.arange(len(classes))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, name in enumerate(names):
        values = [data[name]["per_class"][c]["f1"] for c in classes]
        ax.bar(x + (i - 1.5) * width, values, width, label=name, color=COLORS[i])
    ax.set_xticks(x, [c.title().replace(" Or ", " or ") for c in classes], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test F1")
    ax.set_title("Per-class test F1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "per_class_f1.png", dpi=160)
    plt.close(fig)


def training_curves_figure(data: dict[str, dict]) -> None:
    """Validation macro-F1 per epoch for all four runs."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, metrics) in enumerate(data.items()):
        history = metrics["training_history"]
        ax.plot(
            [h["epoch"] for h in history],
            [h["val_macro_f1"] for h in history],
            label=name,
            color=COLORS[i],
            linewidth=2,
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Validation macro-F1 during training")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "validation_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    """Write the three comparison figures to reports/figures."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = _load()
    overall_figure(data)
    per_class_figure(data)
    training_curves_figure(data)
    print(f"Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
