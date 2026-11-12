"""Offline tests for synthetic narrative validation and augmentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import generate_synthetic as generator  # noqa: E402


class FakeModel:
    """Return queued text responses without making network calls."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.prompts: list[str] = []

    def generate_content(self, prompt: str):
        self.prompts.append(prompt)
        return type("Response", (), {"text": next(self.responses)})()


def test_generation_retries_invalid_responses_and_uses_training_examples(monkeypatch):
    examples = [
        f"real example number {index} from a training incident"
        for index in range(3)
    ]
    valid = "a detailed pipeline incident narrative with pressure " + "technical details " * 25
    model = FakeModel(["TOO SHORT", valid])
    stats: dict[str, int] = {}
    monkeypatch.setattr(generator.time, "sleep", lambda _: None)

    result = generator.generate_synthetic_narratives(
        "EXCAVATION DAMAGE", 1, examples, model, stats=stats
    )

    assert result == [valid.strip()]
    assert len(model.prompts) == 2
    assert all(any(example in prompt for example in examples) for prompt in model.prompts)
    assert stats == {"failed_validation_attempts": 1, "regenerated_narratives": 1}


def _make_label_map(path: Path) -> dict[str, int]:
    label2id = {
        f"CAUSE {index}": index
        for index in range(8)
    }
    path.write_text(json.dumps({"label2id": label2id}), encoding="utf-8")
    return label2id


def test_build_augmented_train_checks_counts_and_saves_source(tmp_path, monkeypatch, capsys):
    label_map_path = tmp_path / "label_map.json"
    label2id = _make_label_map(label_map_path)
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "val.csv"
    test_path = tmp_path / "test.csv"
    synthetic_path = tmp_path / "synthetic.csv"
    output_path = tmp_path / "augmented.csv"

    real_rows = []
    for cause, label in label2id.items():
        count = 299 if cause == "CAUSE 0" else 300
        for index in range(count):
            real_rows.append({
                "REPORT_NUMBER": f"{label}-{index}",
                "CAUSE": cause,
                "narrative_clean": f"real training narrative {cause.lower()} {index}",
                "label": label,
            })
    pd.DataFrame(real_rows).to_csv(train_path, index=False)
    pd.DataFrame({"narrative_clean": ["validation only text"]}).to_csv(validation_path, index=False)
    pd.DataFrame({"narrative_clean": ["test only text"]}).to_csv(test_path, index=False)
    synthetic_text = (
        "synthetic pipeline incident details "
        + "lowercase technical narrative " * 20
    ).strip()
    pd.DataFrame([{
        "narrative_clean": synthetic_text,
        "CAUSE": "CAUSE 0",
        "label": 0,
        "source": "synthetic",
    }]).to_csv(synthetic_path, index=False)
    monkeypatch.setattr(generator, "VALIDATION_CSV", validation_path)
    monkeypatch.setattr(generator, "TEST_CSV", test_path)

    augmented = generator.build_augmented_train(
        train_path, synthetic_path, output_path, label_map_path
    )

    assert len(augmented) == len(real_rows) + 1
    assert augmented["CAUSE"].value_counts().min() >= 300
    assert augmented["source"].value_counts().to_dict() == {
        "real": len(real_rows),
        "synthetic": 1,
    }
    pd.testing.assert_frame_equal(
        pd.read_csv(output_path),
        augmented,
        check_dtype=False,
        check_names=True,
    )
    assert capsys.readouterr().out.count("PASS:") == 6


def test_generation_rejects_cause_name_and_reports_exhausted_retries(monkeypatch):
    examples = [f"example {index}" for index in range(3)]
    model = FakeModel(["bad " + "text " * 30 + "excavation damage"] * 4)
    monkeypatch.setattr(generator.time, "sleep", lambda _: None)

    with pytest.raises(ValueError, match="after four attempts"):
        generator.generate_synthetic_narratives(
            "EXCAVATION DAMAGE", 1, examples, model
        )

    assert len(model.prompts) == 4
