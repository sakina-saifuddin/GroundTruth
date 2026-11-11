"""Build Experiment B's augmented training file from user-supplied synthetic text."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_augmented_train(
    train_csv_path: str | Path,
    label_map_path: str | Path,
    output_path: str | Path,
    synthetic_csv_path: str | Path,
) -> pd.DataFrame:
    """Add supplied synthetic examples to training classes with fewer than 300 rows."""
    train_path = Path(train_csv_path)
    map_path = Path(label_map_path)
    synthetic_path = Path(synthetic_csv_path)
    if not synthetic_path.is_file():
        raise FileNotFoundError(
            f"Synthetic training file is required but missing: {synthetic_path}. "
            "Do not run Experiment B until you create and review this file."
        )

    train = pd.read_csv(train_path)
    synthetic = pd.read_csv(synthetic_path)
    label_data = json.loads(map_path.read_text(encoding="utf-8"))
    label2id = {str(cause): int(label) for cause, label in label_data["label2id"].items()}

    required_train = {"narrative_clean", "CAUSE", "label"}
    required_synthetic = {"narrative_clean", "CAUSE", "label"}
    if not required_train.issubset(train.columns):
        raise ValueError(f"{train_path} must contain {sorted(required_train)}.")
    if not required_synthetic.issubset(synthetic.columns):
        raise ValueError(f"{synthetic_path} must contain {sorted(required_synthetic)}.")
    if train["narrative_clean"].isna().any():
        raise ValueError("The real training split contains missing cleaned narratives.")

    train["label"] = pd.to_numeric(train["label"], errors="raise").astype(int)
    train["CAUSE"] = train["CAUSE"].astype(str).str.strip().str.upper()
    if not train["CAUSE"].isin(label2id).all() or not train["CAUSE"].map(label2id).eq(train["label"]).all():
        raise ValueError("Real training causes and integer labels do not match label_map.json.")
    synthetic["label"] = pd.to_numeric(synthetic["label"], errors="raise").astype(int)
    synthetic["CAUSE"] = synthetic["CAUSE"].astype(str).str.strip().str.upper()
    synthetic["narrative_clean"] = synthetic["narrative_clean"].astype("string")

    valid_ids = set(label2id.values())
    if not synthetic["label"].isin(valid_ids).all():
        raise ValueError("Synthetic rows contain labels outside label_map.json.")
    expected_labels = synthetic["CAUSE"].map(label2id)
    if expected_labels.isna().any() or not expected_labels.eq(synthetic["label"]).all():
        raise ValueError("Every synthetic CAUSE must match its integer label in label_map.json.")
    if synthetic["narrative_clean"].isna().any() or synthetic["narrative_clean"].str.strip().eq("").any():
        raise ValueError("Synthetic narratives must be non-empty text.")
    if not synthetic["narrative_clean"].eq(synthetic["narrative_clean"].str.strip()).all():
        raise ValueError("Synthetic narratives must not have leading or trailing whitespace.")
    if not synthetic["narrative_clean"].eq(synthetic["narrative_clean"].str.lower()).all():
        raise ValueError("Synthetic narratives must already be lowercase.")
    if not synthetic["CAUSE"].isin(label2id).all():
        raise ValueError("Synthetic data contains a cause that is absent from label_map.json.")

    real_counts = train["CAUSE"].value_counts()
    rare_causes = {
        cause for cause in label2id if int(real_counts.get(cause, 0)) < 300
    }

    validation_path = ROOT / "data" / "processed" / "val.csv"
    test_path = ROOT / "data" / "processed" / "test.csv"
    if not validation_path.is_file() or not test_path.is_file():
        raise FileNotFoundError("Phase 1 val.csv and test.csv are required for leakage checks.")
    validation_text = set(pd.read_csv(validation_path)["narrative_clean"].dropna().astype(str))
    test_text = set(pd.read_csv(test_path)["narrative_clean"].dropna().astype(str))
    if synthetic["narrative_clean"].isin(validation_text | test_text).any():
        raise ValueError("A synthetic narrative duplicates a validation or test narrative.")

    real_text = set(train["narrative_clean"].astype(str))
    if synthetic["narrative_clean"].isin(real_text).any():
        raise ValueError("A synthetic narrative duplicates a real training narrative.")
    if synthetic["narrative_clean"].duplicated().any():
        raise ValueError("Synthetic narratives must be unique within synthetic_train.csv.")

    # Keep the Phase 1 training schema while leaving unavailable audit fields empty.
    for column in train.columns:
        if column not in synthetic.columns:
            synthetic[column] = pd.NA
    synthetic["NARRATIVE"] = synthetic["narrative_clean"]
    synthetic = synthetic[train.columns]
    synthetic_rare = synthetic.loc[synthetic["CAUSE"].isin(rare_causes)].copy()
    augmented = pd.concat([train, synthetic_rare.reindex(columns=train.columns)], ignore_index=True)

    augmented_counts = augmented["CAUSE"].value_counts()
    underfilled = {
        cause: int(augmented_counts.get(cause, 0))
        for cause in rare_causes
        if int(augmented_counts.get(cause, 0)) < 300
    }
    if underfilled:
        raise ValueError(
            "Synthetic data did not bring every rare class to at least 300 examples: "
            f"{underfilled}"
        )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(destination, index=False, encoding="utf-8", lineterminator="\n")
    print(f"Rare training causes (<300 before augmentation): {sorted(rare_causes)}")
    print(f"Augmented training rows written: {len(augmented):,} -> {destination}")
    return augmented


def main() -> None:
    """Build the augmented training CSV using paths relative to the repository."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, default=ROOT / "data" / "processed" / "train.csv")
    parser.add_argument("--label-map", type=Path, default=ROOT / "data" / "processed" / "label_map.json")
    parser.add_argument(
        "--synthetic-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "synthetic_train.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "sakina" / "processed" / "augmented_train.csv",
    )
    args = parser.parse_args()
    build_augmented_train(args.train_csv, args.label_map, args.output, args.synthetic_csv)


if __name__ == "__main__":
    main()
