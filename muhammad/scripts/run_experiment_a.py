"""Train and test Experiment A: BiLSTM-Attention on clean training data with class weights."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from muhammad.src.bilstm_attention import BiLSTMAttentionClassifier  # noqa: E402
from muhammad.src.comparison import generate_comparison  # noqa: E402
from muhammad.src.dataset import PHMSASequenceDataset, Vocabulary  # noqa: E402
from muhammad.src.evaluate import evaluate_model, save_results  # noqa: E402
from muhammad.src.train import train_model  # noqa: E402


def main() -> None:
    """Train Experiment A, evaluate its best checkpoint, and save measured scores."""
    config = {
        "experiment_name": "experiment_a",
        "train_csv": str(ROOT / "data" / "experiment1" / "train.csv"),
        "val_csv": str(ROOT / "data" / "experiment1" / "val.csv"),
        "test_csv": str(ROOT / "data" / "experiment1" / "test.csv"),
        "label_map": str(ROOT / "data" / "processed" / "label_map.json"),
        "class_weights": str(ROOT / "data" / "experiment1" / "class_weights.json"),
        "fasttext_path": str(ROOT / "muhammad" / "fasttext" / "model.bin"),
        "freeze_embeddings": True,
        "max_length": 300,
        "max_vocab_size": 50_000,
        "embed_dim": 300,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.3,
        "batch_size": 32,
        "lr": 1e-3,
        "epochs": 30,
        "patience": 5,
        "use_class_weights": True,
        "checkpoint_dir": str(ROOT / "muhammad" / "checkpoints"),
        "results_dir": str(ROOT / "muhammad" / "results" / "experiment_a"),
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
    vocabulary = Vocabulary.load(training["vocab_path"])
    model = BiLSTMAttentionClassifier(
        vocab_size=len(vocabulary),
        embed_dim=config["embed_dim"],
        hidden_size=config["hidden_size"],
        num_labels=config["num_labels"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        freeze_embeddings=config.get("freeze_embeddings", True),
    )
    checkpoint = torch.load(training["checkpoint_path"], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_dataset = PHMSASequenceDataset(
        config["test_csv"], vocabulary, max_length=config["max_length"]
    )
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    results = evaluate_model(model, test_loader, device, label_names)
    results["best_val_f1"] = float(training["best_val_f1"])
    results["best_epoch"] = int(training["best_epoch"])
    results["trainable_parameters"] = int(training["trainable_parameters"])
    results["training_history"] = training["history"]

    save_results(results, config["results_dir"], config["experiment_name"])
    generate_comparison(
        ROOT / "muhammad" / "results",
        ROOT / "muhammad" / "results" / "comparison.md",
    )
    print(
        f"Experiment A complete: best validation macro-F1={training['best_val_f1']:.4f} "
        f"(epoch {training['best_epoch']}), test macro-F1={results['macro_f1']:.4f}."
    )


if __name__ == "__main__":
    main()
