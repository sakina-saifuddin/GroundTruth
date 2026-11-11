"""Tests for the shared, reproducible Phase 1 data pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_data import run_pipeline  # noqa: E402
from src.data_prep import LABELS, clean_narrative  # noqa: E402

WORKBOOK = ROOT / "data" / "raw" / "PHMSA_Gas_Transmission_Gathering_Merged_2002_Present.xlsx"


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Run preparation once into a temporary directory for shared assertions."""
    tmp_path = tmp_path_factory.mktemp("prepared")
    outputs = tmp_path / "processed"
    report = tmp_path / "reports" / "data_report.md"
    figures = tmp_path / "reports" / "figures"
    result = run_pipeline(WORKBOOK, outputs, report, figures)
    return result, outputs, report, figures


def test_cleaning_preserves_pressure_values_and_removes_html_tags():
    """Only real markup is removed; literal engineering comparisons remain."""
    assert clean_narrative("PRESSURE <450PSIG<br>  NOW") == "pressure <450psig now"
    assert clean_narrative("  A_x000D_\tB &amp; C ") == "a b & c"


def test_dataset_size_and_cause_labels(prepared):
    """The cleaned dataset has the stated usable rows and label mapping."""
    result, outputs, _, _ = prepared
    data = pd.read_csv(outputs / "clean_full.csv")
    assert len(data) == 3026
    assert data["NARRATIVE"].notna().all()
    assert data["narrative_clean"].ne("").all()
    assert data["CAUSE"].nunique() == 8
    assert set(data["label"]) == set(range(8))

    label_map = json.loads((outputs / "label_map.json").read_text(encoding="utf-8"))
    assert label_map["label2id"] == {cause: index for index, cause in enumerate(LABELS)}
    assert {int(label): cause for label, cause in label_map["id2label"].items()} == {
        index: cause for index, cause in enumerate(LABELS)
    }


def test_splits_are_disjoint_balanced_and_complete(prepared):
    """No row or narrative leaks, and all classes retain their full-data share."""
    result, outputs, _, _ = prepared
    full = pd.read_csv(outputs / "clean_full.csv")
    splits = {name: pd.read_csv(outputs / f"{name}.csv") for name in ("train", "val", "test")}

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert set(splits[left]["REPORT_NUMBER"]).isdisjoint(splits[right]["REPORT_NUMBER"])
        assert set(splits[left]["narrative_clean"]).isdisjoint(splits[right]["narrative_clean"])

    shares = full["CAUSE"].value_counts(normalize=True)
    for frame in splits.values():
        assert set(frame["CAUSE"]) == set(LABELS)
        current = frame["CAUSE"].value_counts(normalize=True)
        for cause in LABELS:
            assert abs(current[cause] - shares[cause]) <= 0.0100001


def test_training_weights_are_balanced(prepared):
    """Weights are computed only from train and satisfy the balanced-sum identity."""
    result, outputs, _, _ = prepared
    train = pd.read_csv(outputs / "train.csv")
    weight_data = json.loads((outputs / "class_weights_train.json").read_text(encoding="utf-8"))
    weights = weight_data["by_cause"]
    assert weight_data["ordered_by_label"] == [weights[cause] for cause in LABELS]
    weighted_sum = sum(weights[cause] * int(train["CAUSE"].eq(cause).sum()) for cause in LABELS)
    assert weighted_sum == pytest.approx(len(train))


def test_running_pipeline_twice_writes_identical_artifacts(prepared, tmp_path):
    """CSV, JSON, report, and chart artifacts are byte-identical across runs."""
    _, first_outputs, first_report, first_figures = prepared
    second_outputs = tmp_path / "second" / "processed"
    second_report = tmp_path / "second" / "reports" / "data_report.md"
    second_figures = tmp_path / "second" / "reports" / "figures"
    run_pipeline(WORKBOOK, second_outputs, second_report, second_figures)

    for relative in (
        "clean_full.csv",
        "train.csv",
        "val.csv",
        "test.csv",
        "label_map.json",
        "class_weights_train.json",
        "split_manifest.csv",
    ):
        assert (first_outputs / relative).read_bytes() == (second_outputs / relative).read_bytes()
    assert first_report.read_bytes() == second_report.read_bytes()
    for image in first_figures.iterdir():
        assert image.read_bytes() == (second_figures / image.name).read_bytes()
