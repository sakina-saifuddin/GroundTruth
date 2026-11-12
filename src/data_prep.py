"""Functions for cleaning, splitting, and reporting on PHMSA narratives."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
LABELS = [
    "EQUIPMENT FAILURE",
    "CORROSION FAILURE",
    "MATERIAL FAILURE OF PIPE OR WELD",
    "EXCAVATION DAMAGE",
    "NATURAL FORCE DAMAGE",
    "OTHER OUTSIDE FORCE DAMAGE",
    "OTHER INCIDENT CAUSE",
    "INCORRECT OPERATION",
]
SOURCE_COLUMNS = [
    "REPORT_NUMBER",
    "IYEAR",
    "NAME",
    "CAUSE",
    "CAUSE_DETAILS",
    "NARRATIVE",
]
OUTPUT_COLUMNS = SOURCE_COLUMNS + ["narrative_clean", "label"]
EXPECTED_DUPLICATE_PAIRS = [
    (20030099, 20040012),
    (20040078, 20040102),
    (20100078, 20100079),
]
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9]*(\s[^<>]*)?/?>")
WHITESPACE_RE = re.compile(r"(?:_x000D_|\s)+")
REPORT_START = "<!-- BERT_TOKEN_REPORT_START -->"
REPORT_END = "<!-- BERT_TOKEN_REPORT_END -->"


def clean_narrative(text: str) -> str:
    """Normalize narrative formatting while retaining useful words and symbols."""
    if not isinstance(text, str):
        raise TypeError(f"NARRATIVE must be text, got {type(text).__name__}")

    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text)
    # Keep stop words, punctuation, and numbers: pressures and pipe sizes carry meaning.
    return text.strip().lower()


def read_source(input_path: Path) -> pd.DataFrame:
    """Load the raw PHMSA CSV. Returns the full dataframe."""
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Required input file not found: {input_path}. "
            "Place PHMSA_Raw_Unclean_2002_Present.csv at data/raw/."
        )
    return pd.read_csv(input_path, encoding="utf-8", low_memory=False)


def _require(condition: bool, message: str) -> None:
    """Stop preparation when a stated source-data fact is not true."""
    if not condition:
        raise ValueError(f"Source data check failed: {message}")


def _check_source_fact(
    condition: bool,
    name: str,
    details: str,
    *,
    fatal: bool = True,
) -> None:
    """Print a source-data check and stop only when the failed check is critical."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}: {details}")
    if fatal and not condition:
        raise ValueError(f"Source data check failed: {name}: {details}")


def validate_source_facts(raw: pd.DataFrame) -> pd.DataFrame:
    """Verify the stated raw-CSV facts and return all usable rows unchanged."""
    columns_match = list(raw.columns) == SOURCE_COLUMNS
    _check_source_fact(
        columns_match and raw.shape == (3044, 6),
        "CSV shape and columns",
        f"expected (3044, 6) and {SOURCE_COLUMNS}; got {raw.shape} and {list(raw.columns)}",
    )

    blank_narrative = raw["NARRATIVE"].isna() | raw["NARRATIVE"].astype("string").str.strip().eq("")
    empty_count = int(blank_narrative.sum())
    _check_source_fact(
        empty_count == 0,
        "Empty narratives",
        f"expected 0; found {empty_count}",
    )

    years = pd.to_numeric(raw["IYEAR"], errors="raise")
    early = raw.loc[years.eq(2001)]
    early_report_ok = (
        len(early) == 1
        and int(early.iloc[0]["REPORT_NUMBER"]) == 20040073
        and str(early.iloc[0]["NAME"]).strip().upper().startswith("AQUILA SOUTHWEST")
        and str(early.iloc[0]["CAUSE"]).strip().upper() == "EXCAVATION DAMAGE"
    )
    early_narrative = str(early.iloc[0]["NARRATIVE"]).lower() if len(early) == 1 else ""
    dates_ok = "11-7-2001" in early_narrative and "1-21-02" in early_narrative
    year_ok = years.min() == 2001 and years.max() == 2026 and early_report_ok and dates_ok
    _check_source_fact(
        year_ok,
        "Year range and 2001 incident",
        f"range {years.min()}-{years.max()}; 2001 reports {early['REPORT_NUMBER'].tolist()}; "
        f"operator {early.iloc[0]['NAME'] if len(early) == 1 else 'missing'}; "
        "expected report 20040073 (AQUILA SOUTHWEST, EXCAVATION DAMAGE) with 11-7-2001 "
        "incident and 1-21-02 discovery dates",
    )

    year_2026_count = int(years.eq(2026).sum())
    _check_source_fact(
        year_2026_count == 88,
        "Partial 2026 snapshot",
        f"expected 88; found {year_2026_count}",
    )

    causes = raw["CAUSE"].dropna().astype(str)
    _check_source_fact(
        causes.nunique() == 8 and set(causes) == set(LABELS),
        "Cause labels",
        f"found {causes.nunique()} distinct values; exact label-set match: {set(causes) == set(LABELS)}",
    )

    report_numbers_unique = raw["REPORT_NUMBER"].is_unique
    _check_source_fact(
        report_numbers_unique,
        "Unique report numbers",
        f"duplicate count: {int(raw['REPORT_NUMBER'].duplicated().sum())}",
    )

    duplicate_groups = raw.groupby("NARRATIVE", sort=False)["REPORT_NUMBER"].apply(list)
    duplicate_groups = [sorted(map(int, group)) for group in duplicate_groups if len(group) > 1]
    expected_groups = [sorted(pair) for pair in EXPECTED_DUPLICATE_PAIRS]
    _check_source_fact(
        sorted(duplicate_groups) == sorted(expected_groups),
        "Duplicate narrative pairs",
        f"expected {expected_groups}; found {duplicate_groups}",
    )

    narratives = raw["NARRATIVE"].astype(str)
    uppercase_count = int(narratives.map(lambda text: text == text.upper()).sum())
    _check_source_fact(
        uppercase_count == len(raw),
        "Narratives are all uppercase",
        f"{uppercase_count}/{len(raw)} are uppercase; {len(raw) - uppercase_count} are not",
        fatal=False,
    )

    after_unescape = narratives.map(html.unescape)
    after_tags = after_unescape.map(lambda text: HTML_TAG_RE.sub(" ", text))
    after_whitespace = after_tags.map(lambda text: WHITESPACE_RE.sub(" ", text).strip())
    whitespace_rows = int(after_tags.ne(after_whitespace).sum())
    _check_source_fact(
        whitespace_rows > 0,
        "Narratives contain extra whitespace",
        f"whitespace normalization will change {whitespace_rows} rows",
    )

    return raw.copy().reset_index(drop=True)


def clean_dataset(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean narratives and labels, recording how many rows each step changed."""
    data = raw.copy()
    original = data["NARRATIVE"].astype(str)
    after_unescape = original.map(html.unescape)
    after_tags = after_unescape.map(lambda text: HTML_TAG_RE.sub(" ", text))
    after_whitespace = after_tags.map(lambda text: WHITESPACE_RE.sub(" ", text).strip())
    after_lower = after_whitespace.str.lower()

    changed = {
        "HTML entity unescaping": int(original.ne(after_unescape).sum()),
        "HTML tag removal": int(after_unescape.ne(after_tags).sum()),
        "Whitespace normalization": int(after_tags.ne(after_whitespace).sum()),
        "lowercasing": int(after_whitespace.ne(after_lower).sum()),
        "Total rows changed by any cleaning": int(original.ne(after_lower).sum()),
    }

    data["narrative_clean"] = data["NARRATIVE"].map(clean_narrative)
    data["CAUSE"] = data["CAUSE"].astype(str).str.strip().str.upper()
    _require(data["CAUSE"].nunique() == 8 and set(data["CAUSE"]) == set(LABELS), "cleaned labels are not the 8 expected classes")
    _require(data["narrative_clean"].ne("").all(), "cleaning produced an empty narrative")
    label2id = {label: index for index, label in enumerate(LABELS)}
    data["label"] = data["CAUSE"].map(label2id).astype("int64")
    data = data.sort_values("REPORT_NUMBER", kind="stable").reset_index(drop=True)
    return data, changed


def make_splits(data: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Create reproducible stratified splits without separating duplicate narratives."""
    group_labels = data[["narrative_clean", "CAUSE"]].drop_duplicates()
    _require(
        not group_labels["narrative_clean"].duplicated().any(),
        "a narrative group has multiple cause labels",
    )
    group_ids = group_labels["narrative_clean"].to_numpy()
    group_causes = group_labels["CAUSE"].to_numpy()

    train_groups, temp_groups, train_causes, temp_causes = train_test_split(
        group_ids,
        group_causes,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=group_causes,
    )
    val_groups, test_groups = train_test_split(
        temp_groups,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_causes,
    )
    assignments = {
        **{group: "train" for group in train_groups},
        **{group: "val" for group in val_groups},
        **{group: "test" for group in test_groups},
    }
    data = data.copy()
    data["split"] = data["narrative_clean"].map(assignments)
    _require(data["split"].notna().all(), "some rows did not receive a split")

    splits = {
        split: data.loc[data["split"].eq(split), OUTPUT_COLUMNS].copy().reset_index(drop=True)
        for split in ("train", "val", "test")
    }
    manifest = data[["REPORT_NUMBER", "split"]].copy()
    validate_splits(data, splits)
    return splits, manifest


def validate_splits(data: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> None:
    """Check leakage, class coverage, and class-share balance in every split."""
    report_sets = [set(frame["REPORT_NUMBER"]) for frame in splits.values()]
    narrative_sets = [set(frame["narrative_clean"]) for frame in splits.values()]
    _require(
        all(report_sets[i].isdisjoint(report_sets[j]) for i in range(3) for j in range(i + 1, 3)),
        "REPORT_NUMBER leakage exists between splits",
    )
    _require(
        all(narrative_sets[i].isdisjoint(narrative_sets[j]) for i in range(3) for j in range(i + 1, 3)),
        "narrative leakage exists between splits",
    )

    full_shares = data["CAUSE"].value_counts(normalize=True)
    for split_name, frame in splits.items():
        _require(set(frame["CAUSE"]) == set(LABELS), f"{split_name} does not contain all 8 classes")
        split_shares = frame["CAUSE"].value_counts(normalize=True)
        for cause in LABELS:
            _require(
                abs(float(split_shares[cause] - full_shares[cause])) <= 0.0100001,
                f"{split_name} share for {cause} differs by more than 1 percentage point",
            )


def calculate_class_weights(train: pd.DataFrame) -> tuple[dict[str, float], list[float]]:
    """Compute balanced class weights using training rows only."""
    counts = train["CAUSE"].value_counts()
    _require(set(counts.index) == set(LABELS), "training split is missing a class")
    total = len(train)
    by_cause = {cause: total / (len(LABELS) * int(counts[cause])) for cause in LABELS}
    ordered = [by_cause[cause] for cause in LABELS]
    return by_cause, ordered


def write_json(path: Path, value: Any) -> None:
    """Write stable, human-readable JSON for reproducible artifacts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def class_distribution_table(data: pd.DataFrame, splits: dict[str, pd.DataFrame]) -> str:
    """Format counts and shares for all classes and data partitions."""
    frames = {"Full": data, "Train": splits["train"], "Validation": splits["val"], "Test": splits["test"]}
    headers = ["Cause"] + [f"{name} count" for name in frames] + [f"{name} share" for name in frames]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for cause in LABELS:
        counts = [int(frame["CAUSE"].eq(cause).sum()) for frame in frames.values()]
        shares = [count / len(frame) * 100 for count, frame in zip(counts, frames.values())]
        cells = [cause, *map(str, counts), *(f"{share:.2f}%" for share in shares)]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_base_report(
    raw_rows: int,
    uppercase_rows: int,
    data: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    changed: dict[str, int],
    weights: dict[str, float],
    word_stats: dict[str, float | int],
    duplicate_group_count: int,
    existing_report: Path | None = None,
) -> str:
    """Build the computed data report while preserving a prior token report."""
    bert_section = "Not run yet. Run `python scripts/token_length_report.py` to add WordPiece statistics."
    if existing_report and existing_report.exists():
        previous = existing_report.read_text(encoding="utf-8")
        if REPORT_START in previous and REPORT_END in previous:
            bert_section = previous.split(REPORT_START, 1)[1].split(REPORT_END, 1)[0].strip()

    count_text = " / ".join(
        f"{name}: {len(splits[name]):,}" for name in ("train", "val", "test")
    )
    weight_rows = "\n".join(
        f"| {cause} | {int(splits['train']['CAUSE'].eq(cause).sum())} | {weight:.12g} |"
        for cause, weight in weights.items()
    )
    word_stat_rows = "\n".join(
        "| {} | {} |".format(
            label,
            value if isinstance(value, int) else f"{value:.2f}",
        )
        for label, value in word_stats.items()
    )
    report = f"""# Shared data preparation report

This report is generated by `python scripts/prepare_data.py`. The modeling input is `narrative_clean`; `NAME` and `CAUSE_DETAILS` are retained for auditing but are not model inputs.

## 1. Source and row counts

- Source: raw public PHMSA Gas Transmission and Gathering Incident Data CSV.
- Rows read: {raw_rows:,}.
- Empty narratives removed: 0.
- Usable rows: {len(data):,}.
- Narratives fully uppercase in the source: {uppercase_rows:,} of {raw_rows:,}; the remaining {raw_rows - uppercase_rows:,} are also lowercased during cleaning.

## 2. Years

- Year range: {int(data['IYEAR'].min())}–{int(data['IYEAR'].max())}.
- The sole 2001 record is REPORT_NUMBER 20040073 (AQUILA SOUTHWEST, EXCAVATION DAMAGE). Its narrative states the pipe was hit on 11-7-2001 and discovered on 1-21-02 (January 21, 2002), so it remains in the dataset.
- 2026 is a partial-year snapshot; it contains {int(data['IYEAR'].eq(2026).sum())} records in this CSV.

## 3. Cleaning

HTML entities are unescaped; only real HTML tags matching the requested tag pattern are replaced; Excel `_x000D_` artifacts and whitespace runs are normalized; surrounding spaces are stripped; text is lowercased. Stop words, punctuation, and numbers are retained.

| Cleaning step | Rows changed |
|---|---:|
"""
    report += "\n".join(f"| {step} | {count:,} |" for step, count in changed.items())
    report += f"""

## 4. Duplicate narratives and split handling

- Duplicate narrative groups: {duplicate_group_count}.
- All identical cleaned narratives are assigned to a single split; no narrative text or report number crosses train, validation, and test.
- Split sizes: {count_text}.

## 5. Class distribution

{class_distribution_table(data, splits)}

## 6. Training class weights

Weights use training rows only and follow `n_train / (8 * class_count)`.

| Cause | Train count | Weight |
|---|---:|---:|
{weight_rows}

## 7. Narrative word lengths

Word counts use whitespace splitting on `narrative_clean`.

| Statistic | Words |
|---|---:|
{word_stat_rows}

## 8. BERT WordPiece token lengths

{REPORT_START}
{bert_section}
{REPORT_END}

## 9. Lecture concepts used

| Concept | Course connection |
|---|---|
| Regex-based narrative cleaning | Lecture 2 |
| Tokenization and WordPiece | Lectures 3 and 13 |
| Stratified train/dev/test split and class imbalance | Lectures 5–6 |
| Class-weighted cross-entropy | Lectures 6 and 9 |
"""
    return report


def compute_word_statistics(data: pd.DataFrame) -> dict[str, float | int]:
    """Compute requested descriptive statistics for narrative word counts."""
    lengths = data["narrative_clean"].str.split().str.len()
    return {
        "Minimum": int(lengths.min()),
        "25th percentile": float(lengths.quantile(0.25)),
        "Median": float(lengths.median()),
        "Mean": float(lengths.mean()),
        "75th percentile": float(lengths.quantile(0.75)),
        "Maximum": int(lengths.max()),
        "Narratives under 10 words": int(lengths.lt(10).sum()),
        "Narratives over 380 words": int(lengths.gt(380).sum()),
    }
