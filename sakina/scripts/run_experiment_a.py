"""Train and test Experiment A: frozen BERT with training class weights."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sakina.src.bert_dataset import PHMSADataset  # noqa: E402
from sakina.src.bert_model import FrozenBertClassifier  # noqa: E402
from sakina.src.comparison import generate_comparison  # noqa: E402
from sakina.src.evaluate import evaluate_model, save_results  # noqa: E402
from sakina.src.train import train_model  # noqa: E402


def main() -> None:
    """Train Experiment A, evaluate its best checkpoint, and save measured scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pooling",
        choices=("mean", "pooler"),
        default="mean",
        help="mean = version 2 (results_mean_pooling); pooler = version 1 (results).",
    )
    pooling = parser.parse_args().pooling
    results_root = ROOT / "sakina" / ("results_mean_pooling" if pooling == "mean" else "results")
    config = {
        "pooling": pooling,
        "experiment_name": "experiment_a",
        "train_csv": str(ROOT / "data" / "processed" / "train.csv"),
        "val_csv": str(ROOT / "data" / "processed" / "val.csv"),
        "test_csv": str(ROOT / "data" / "processed" / "test.csv"),
        "label_map": str(ROOT / "data" / "processed" / "label_map.json"),
        "class_weights": str(ROOT / "data" / "processed" / "class_weights_train.json"),
        "model_name": "bert-base-uncased",
        "max_length": 512,
        "batch_size": 16,
        "lr": 2e-3,
        "epochs": 20,
        "patience": 5,
        "use_class_weights": True,
        "checkpoint_dir": str(ROOT / "sakina" / "checkpoints"),
        "results_dir": str(results_root / "experiment_a"),
        "seed": 42,
        "num_labels": 8,
    }
    for key in ("train_csv", "val_csv", "test_csv", "label_map", "class_weights"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"Required Experiment A input not found: {config[key]}")

    labels = json.loads(Path(config["label_map"]).read_text(encoding="utf-8"))
    label_names = [labels["id2label"][str(index)] for index in range(config["num_labels"])]
    training = train_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenBertClassifier(
        num_labels=config["num_labels"],
        model_name=config["model_name"],
        pooling=pooling,
    )
    checkpoint = torch.load(training["checkpoint_path"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    test_dataset = PHMSADataset(config["test_csv"], tokenizer, max_length=config["max_length"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    results = evaluate_model(model, test_loader, device, label_names)
    results["best_val_f1"] = float(training["best_val_f1"])
    results["best_epoch"] = int(training["best_epoch"])
    results["trainable_parameters"] = int(training["trainable_parameters"])
    results["training_history"] = training["history"]
    save_results(results, config["results_dir"], config["experiment_name"])
    generate_comparison(
        results_root,
        results_root / "comparison.md",
    )
    print(
        f"Experiment A complete: best validation macro-F1={training['best_val_f1']:.4f} "
        f"(epoch {training['best_epoch']}), test macro-F1={results['macro_f1']:.4f}."
    )


if __name__ == "__main__":
    main()
