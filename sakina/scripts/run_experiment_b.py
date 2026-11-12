"""Train and test Experiment B: oversampled training data without class weights.

Reads the finished Experiment 2 files in data/experiment2/ (train.csv already holds
the real training rows plus the validated synthetic rows)."""

from __future__ import annotations

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
    """Train Experiment B on data/experiment2 and evaluate its test set."""
    config = {
        "experiment_name": "experiment_b",
        "train_csv": str(ROOT / "data" / "experiment2" / "train.csv"),
        "val_csv": str(ROOT / "data" / "experiment2" / "val.csv"),
        "test_csv": str(ROOT / "data" / "experiment2" / "test.csv"),
        "label_map": str(ROOT / "data" / "processed" / "label_map.json"),
        "model_name": "bert-base-uncased",
        "max_length": 512,
        "batch_size": 16,
        "lr": 2e-3,
        "epochs": 20,
        "patience": 5,
        "use_class_weights": False,
        "checkpoint_dir": str(ROOT / "sakina" / "checkpoints"),
        "results_dir": str(ROOT / "sakina" / "results_mean_pooling" / "experiment_b"),
        "seed": 42,
        "num_labels": 8,
    }
    for key in ("train_csv", "val_csv", "test_csv", "label_map"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"Required Experiment B input not found: {config[key]}")

    labels = json.loads(Path(config["label_map"]).read_text(encoding="utf-8"))
    label_names = [labels["id2label"][str(index)] for index in range(config["num_labels"])]
    training = train_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FrozenBertClassifier(num_labels=config["num_labels"], model_name=config["model_name"])
    checkpoint = torch.load(training["checkpoint_path"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    test_dataset = PHMSADataset(tokenizer=tokenizer, dataset_path=config["test_csv"], max_length=config["max_length"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    results = evaluate_model(model, test_loader, device, label_names)
    results["best_val_f1"] = float(training["best_val_f1"])
    results["best_epoch"] = int(training["best_epoch"])
    results["trainable_parameters"] = int(training["trainable_parameters"])
    results["training_history"] = training["history"]
    save_results(results, config["results_dir"], config["experiment_name"])
    generate_comparison(
        ROOT / "sakina" / "results_mean_pooling",
        ROOT / "sakina" / "results_mean_pooling" / "comparison.md",
    )
    print(
        f"Experiment B complete: best validation macro-F1={training['best_val_f1']:.4f} "
        f"(epoch {training['best_epoch']}), test macro-F1={results['macro_f1']:.4f}."
    )


if __name__ == "__main__":
    main()
