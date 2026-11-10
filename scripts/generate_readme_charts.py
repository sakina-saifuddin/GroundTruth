#!/usr/bin/env python3
"""Generate the README charts from the supplied PHMSA incident CSV."""

from __future__ import annotations

import argparse
import csv
import html
import re
from collections import Counter
from pathlib import Path
from statistics import median as median_value

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "images"
REQUIRED_COLUMNS = {"IYEAR", "CAUSE", "NARRATIVE"}
CAUSE_ORDER = [
    "EQUIPMENT FAILURE",
    "CORROSION FAILURE",
    "MATERIAL FAILURE OF PIPE OR WELD",
    "EXCAVATION DAMAGE",
    "NATURAL FORCE DAMAGE",
    "OTHER OUTSIDE FORCE DAMAGE",
    "OTHER INCIDENT CAUSE",
    "INCORRECT OPERATION",
]
EARTH = "#5f3b2e"
CLAY = "#b66a44"
SAND = "#e7c98a"
SAGE = "#71866a"
INK = "#26342b"
PAPER = "#fbf7ef"


def clean_narrative(text: str) -> str:
    text = re.sub(r"<[^>]*>", " ", html.unescape(text))
    return " ".join(text.split())


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        actual_columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - actual_columns
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV is missing required columns: {missing}")
        rows = list(reader)

    if not rows:
        raise ValueError("CSV contains no data rows.")

    for row_number, row in enumerate(rows, start=2):
        if not (row.get("CAUSE") or "").strip():
            raise ValueError(f"Row {row_number} has no CAUSE value.")
        if not (row.get("NARRATIVE") or "").strip():
            raise ValueError(f"Row {row_number} has no NARRATIVE value.")
        try:
            int(row["IYEAR"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Row {row_number} has an invalid IYEAR value.") from error

    return rows


def finish_figure(figure: plt.Figure, output_path: Path) -> None:
    figure.savefig(output_path, dpi=100, facecolor=PAPER, bbox_inches="tight")
    plt.close(figure)
    with Image.open(output_path) as source:
        image = source.convert("RGB")
        if image.width > 1200:
            height = round(image.height * 1200 / image.width)
            image = image.resize((1200, height), Image.Resampling.LANCZOS)
        image.save(output_path, format="PNG", optimize=True, compress_level=9)


def plot_causes(rows: list[dict[str, str]]) -> None:
    counts = Counter(row["CAUSE"].strip() for row in rows)
    order = [cause for cause in CAUSE_ORDER if cause in counts]
    order.extend(sorted(set(counts) - set(order)))
    labels = [cause.title() for cause in order]
    values = [counts[cause] for cause in order]

    figure, axis = plt.subplots(figsize=(11, 6))
    bars = axis.barh(labels[::-1], values[::-1], color=CLAY, edgecolor=EARTH, linewidth=0.6)
    axis.set_title("Narratives by reported cause", loc="left", color=INK, fontsize=18, pad=16)
    axis.set_xlabel("Narratives")
    axis.set_ylabel("")
    axis.grid(axis="x", color="#ded7c9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#b9ad9c")
    axis.bar_label(bars, padding=4, color=INK, fontsize=9)
    figure.tight_layout()
    finish_figure(figure, OUTPUT_DIR / "chart-cause-counts.png")


def plot_years(rows: list[dict[str, str]]) -> None:
    counts = Counter(int(row["IYEAR"]) for row in rows)
    years = sorted(counts)
    values = [counts[year] for year in years]

    figure, axis = plt.subplots(figsize=(11, 6))
    bars = axis.bar(years, values, color=SAGE, edgecolor=INK, linewidth=0.4)
    axis.set_title("Narratives by incident year", loc="left", color=INK, fontsize=18, pad=16)
    axis.set_xlabel("Incident year")
    axis.set_ylabel("Narratives")
    axis.set_xticks(years)
    axis.tick_params(axis="x", labelrotation=45, labelsize=8)
    axis.grid(axis="y", color="#ded7c9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#b9ad9c")
    if len(bars) <= 16:
        axis.bar_label(bars, padding=2, fontsize=8)
    figure.tight_layout()
    finish_figure(figure, OUTPUT_DIR / "chart-incidents-by-year.png")


def plot_lengths(rows: list[dict[str, str]]) -> tuple[float, float]:
    lengths = [word_count(clean_narrative(row["NARRATIVE"])) for row in rows]
    median = float(median_value(lengths))
    mean = sum(lengths) / len(lengths)

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.hist(lengths, bins=32, color=SAND, edgecolor=EARTH, linewidth=0.7)
    axis.axvline(median, color=CLAY, linestyle="--", linewidth=2, label=f"Median: {median:g} words")
    axis.set_title("Narrative length", loc="left", color=INK, fontsize=18, pad=16)
    axis.set_xlabel("Words per narrative")
    axis.set_ylabel("Narratives")
    axis.grid(axis="y", color="#ded7c9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#b9ad9c")
    axis.legend(frameon=False)
    figure.tight_layout()
    finish_figure(figure, OUTPUT_DIR / "chart-narrative-lengths.png")
    return median, mean


def plot_split(row_count: int) -> tuple[int, int, int]:
    train = round(row_count * 0.70)
    validation = round(row_count * 0.15)
    test = row_count - train - validation
    labels = ["Train", "Validation", "Test"]
    values = [train, validation, test]

    figure, axis = plt.subplots(figsize=(11, 6))
    bars = axis.bar(labels, values, color=[EARTH, CLAY, SAND], edgecolor=INK, linewidth=0.6)
    axis.set_title("Planned stratified split", loc="left", color=INK, fontsize=18, pad=16)
    axis.set_ylabel("Narratives")
    axis.grid(axis="y", color="#ded7c9", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color("#b9ad9c")
    axis.bar_label(bars, padding=4, color=INK)
    figure.tight_layout()
    finish_figure(figure, OUTPUT_DIR / "chart-data-split.png")
    return train, validation, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Path to the merged PHMSA CSV.")
    args = parser.parse_args()
    csv_path = args.csv.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows = load_rows(csv_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_causes(rows)
    plot_years(rows)
    median, mean = plot_lengths(rows)
    train, validation, test = plot_split(len(rows))

    print(f"Rows: {len(rows)}")
    print(f"Word count (cleaned narratives): median={median:g}, mean={mean:.1f}")
    print(f"Split sizes (70/15/15): train={train}, validation={validation}, test={test}")
    for image_path in sorted(OUTPUT_DIR.glob("chart-*.png")):
        with Image.open(image_path) as image:
            if image.width > 1200:
                raise RuntimeError(f"Generated image exceeds 1200 px width: {image_path}")
            print(f"Wrote {image_path.relative_to(ROOT)} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()
