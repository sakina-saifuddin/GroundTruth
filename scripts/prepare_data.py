"""Run the shared PHMSA data-preparation and exploratory-analysis pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep import (  # noqa: E402
    LABELS,
    OUTPUT_COLUMNS,
    RANDOM_SEED,
    calculate_class_weights,
    clean_dataset,
    compute_word_statistics,
    generate_base_report,
    make_splits,
    read_source,
    validate_source_facts,
    write_json,
)


def save_csv(frame: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    """Write a stable UTF-8 CSV without an index or platform-specific line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is not None:
        frame = frame[columns]
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def create_figures(
    data: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    figures_dir: Path,
) -> None:
    """Create the three requested dataset overview charts."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {"train": "#477a68", "val": "#d99b42", "test": "#6d83a6"}

    counts = pd.DataFrame(
        {
            split: [int(frame["CAUSE"].eq(cause).sum()) for cause in LABELS]
            for split, frame in splits.items()
        },
        index=LABELS,
    )
    axis = counts.plot(kind="bar", figsize=(12, 6), color=[colors[name] for name in counts.columns])
    axis.set_title("Cause counts by data split")
    axis.set_xlabel("Cause")
    axis.set_ylabel("Incident narratives")
    axis.tick_params(axis="x", labelrotation=35)
    axis.legend(title="Split")
    axis.figure.tight_layout()
    axis.figure.savefig(figures_dir / "class_distribution_by_split.png", dpi=160)
    plt.close(axis.figure)

    lengths = data["narrative_clean"].str.split().str.len()
    median = float(lengths.median())
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(lengths, bins=40, color="#477a68", edgecolor="white")
    axis.axvline(median, color="#b6533c", linestyle="--", label=f"Median: {median:.0f} words")
    axis.set_title("Narrative length distribution")
    axis.set_xlabel("Words per narrative")
    axis.set_ylabel("Narratives")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "narrative_length_words.png", dpi=160)
    plt.close(figure)

    years = data["IYEAR"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(years.index.astype(int), years.values, color="#6d83a6")
    axis.set_title("Incident narratives by year")
    axis.set_xlabel("Incident year")
    axis.set_ylabel("Narratives")
    axis.set_xticks(years.index.astype(int))
    axis.tick_params(axis="x", labelrotation=45)
    figure.tight_layout()
    figure.savefig(figures_dir / "incidents_by_year.png", dpi=160)
    plt.close(figure)


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    report_path: Path,
    figures_dir: Path,
) -> dict[str, object]:
    """Validate, clean, split, and summarize the raw source CSV."""
    raw = read_source(input_path)
    usable_raw = validate_source_facts(raw)
    data, cleaning_changes = clean_dataset(usable_raw)
    expected_cleaning_changes = {
        "HTML entity unescaping": 5,
        "HTML tag removal": 3,
        "Whitespace normalization": 2519,
        "lowercasing": 3044,
        "Total rows changed by any cleaning": 3044,
    }
    for step, expected in expected_cleaning_changes.items():
        actual = cleaning_changes[step]
        status = "PASS" if actual == expected else "FAIL"
        print(f"[{status}] Cleaning {step}: expected {expected:,}; found {actual:,}")
        if actual != expected:
            raise ValueError(
                f"Raw CSV cleaning check failed for {step}: expected {expected}, found {actual}."
            )
    splits, manifest = make_splits(data)
    weights_by_cause, weights_by_label = calculate_class_weights(splits["train"])

    output_dir.mkdir(parents=True, exist_ok=True)
    save_csv(data, output_dir / "clean_full.csv", OUTPUT_COLUMNS)
    for split_name, frame in splits.items():
        save_csv(frame, output_dir / f"{split_name}.csv", OUTPUT_COLUMNS)
    save_csv(manifest, output_dir / "split_manifest.csv", ["REPORT_NUMBER", "split"])

    label2id = {label: index for index, label in enumerate(LABELS)}
    id2label = {str(index): label for label, index in label2id.items()}
    write_json(
        output_dir / "label_map.json",
        {"label2id": label2id, "id2label": id2label},
    )
    write_json(
        output_dir / "class_weights_train.json",
        {
            "by_cause": weights_by_cause,
            "ordered_by_label": weights_by_label,
            "label_order": LABELS,
        },
    )

    create_figures(data, splits, figures_dir)
    word_stats = compute_word_statistics(data)
    duplicate_count = int(data["narrative_clean"].duplicated(keep=False).sum() // 2)
    report = generate_base_report(
        raw_rows=len(raw),
        uppercase_rows=int(raw["NARRATIVE"].astype(str).map(lambda text: text == text.upper()).sum()),
        data=data,
        splits=splits,
        changed=cleaning_changes,
        weights=weights_by_cause,
        word_stats=word_stats,
        duplicate_group_count=duplicate_count,
        existing_report=report_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print("\nSplit sizes:")
    for split_name, frame in splits.items():
        print(f"  {split_name}: {len(frame):,}")
        print(f"    class counts: {json.dumps(frame['CAUSE'].value_counts().reindex(LABELS).to_dict())}")
    print("\nTraining class weights:")
    for cause, weight in weights_by_cause.items():
        print(f"  {cause}: {weight:.6f}")
    print(f"\nFixed random seed: {RANDOM_SEED}")
    print(f"Outputs written to {output_dir} and {report_path.parent}.")
    return {
        "data": data,
        "splits": splits,
        "weights": weights_by_cause,
        "cleaning_changes": cleaning_changes,
    }


def main() -> None:
    """Parse paths and run the complete reproducible preparation pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "raw" / "PHMSA_Raw_Unclean_2002_Present.csv",
        help="Path to the raw PHMSA CSV.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "data_report.md")
    parser.add_argument("--figures-dir", type=Path, default=ROOT / "reports" / "figures")
    args = parser.parse_args()
    run_pipeline(args.input, args.output_dir, args.report, args.figures_dir)


if __name__ == "__main__":
    main()
