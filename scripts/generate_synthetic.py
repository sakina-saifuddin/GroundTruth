"""Generate training-only synthetic PHMSA narratives and build Experiment 2 data."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TARGET_PER_CLASS = 300
GENERATION_MODEL = "gemini-3.5-flash"
GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_PROVIDER = "groq"
# Short pause between calls; rate-limit errors are handled by backoff below.
REQUEST_INTERVAL_SECONDS = 2
MAX_API_RETRIES = 8
TRANSIENT_ERROR_NAMES = {"ResourceExhausted", "DeadlineExceeded", "ServiceUnavailable", "InternalServerError",
    "RateLimitError", "APITimeoutError", "APIConnectionError"}
SYNTHETIC_COLUMNS =["narrative_clean", "CAUSE", "label", "source"]
VALIDATION_CSV = ROOT / "data" / "experiment2" / "val.csv"
TEST_CSV = ROOT / "data" / "experiment2" / "test.csv"


def _prompt_for_narrative(cause: str, examples: list[str]) -> str:
    """Create the requested one-narrative prompt from training examples only."""
    examples_text = "\n---\n".join(examples)
    return f"""You are generating training data for an NLP pipeline incident classifier.

Cause category: {cause}

Here are 3 real examples of narratives from this category:
---
{examples_text}
---

Write 1 new realistic pipeline incident narrative for the cause category "{cause}".
Rules:
- Write in the same style as the examples above
- Similar length (100-300 words)
- All lowercase
- Use realistic pipeline technical details (pressures, pipe sizes, materials)
- Do NOT copy sentences word for word from the examples
- Do NOT include any cause label or category name in the narrative
- Return ONLY the narrative text, nothing else

Narrative:"""


def _is_valid_narrative(
    narrative: str,
    cause: str,
    real_narratives: set[str],
    generated_narratives: set[str],
) -> bool:
    """Reject malformed text, label leakage, and exact duplicates."""
    cleaned = narrative.strip()
    normalized = cleaned.lower()
    return (
        len(cleaned.split()) >= 50
        and cleaned == normalized
        and cause.lower() not in normalized
        and normalized not in real_narratives
        and normalized not in generated_narratives
    )


def _generate_with_backoff(model: Any, prompt: str) -> Any:
    """Call Gemini, waiting and retrying on rate-limit or transient errors."""
    for retry in range(MAX_API_RETRIES + 1):
        try:
            return model.generate_content(prompt, request_options={"timeout": 120})
        except Exception as error:  # noqa: BLE001 - classify by name to avoid hard dependency
            transient = type(error).__name__ in TRANSIENT_ERROR_NAMES
            if not transient or retry == MAX_API_RETRIES:
                raise
            match = re.search(r"(?:retry|try again) in ([\d.]+)s", str(error))
            wait = float(match.group(1)) + 2 if match else 30.0
            print(f"  {type(error).__name__}; waiting {wait:.0f}s (retry {retry + 1}/{MAX_API_RETRIES})")
            time.sleep(wait)


def generate_synthetic_narratives(
    cause: str,
    n: int,
    real_examples: list[str],
    model: Any,
    *,
    stats: dict[str, int] | None = None,
    existing: list[str] | None = None,
    on_narrative: Callable[[str], None] | None = None,
) -> list[str]:
    """Generate validated narratives using examples from the training split only.

    The real_examples argument must contain only narratives from the real
    training split. API failures are surfaced; invalid responses are retried up
    to three times after the initial request.
    """
    if n < 0:
        raise ValueError("n must be non-negative.")
    if n == 0:
        return []
    if len(real_examples) < 3:
        raise ValueError(f"At least three real training examples are required for {cause}.")
    if any(not isinstance(text, str) or not text.strip() for text in real_examples):
        raise ValueError("Real training examples must be non-empty strings.")

    rng = random.Random(SEED)
    normalized_real = {text.strip().lower() for text in real_examples}
    generated: list[str] = []
    generated_normalized: set[str] = {text.strip().lower() for text in existing or []}
    if stats is not None:
        stats.setdefault("failed_validation_attempts", 0)
        stats.setdefault("regenerated_narratives", 0)

    for _ in range(n):
        for attempt in range(4):
            examples = rng.sample(real_examples, 3)
            response = _generate_with_backoff(model, _prompt_for_narrative(cause, examples))
            time.sleep(REQUEST_INTERVAL_SECONDS)
            narrative = response.text.strip() if response.text else ""

            if _is_valid_narrative(
                narrative, cause, normalized_real, generated_normalized
            ):
                if attempt and stats is not None:
                    stats["regenerated_narratives"] += 1
                generated.append(narrative)
                generated_normalized.add(narrative.lower())
                if on_narrative is not None:
                    on_narrative(narrative)
                break

            if stats is not None:
                stats["failed_validation_attempts"] += 1
        else:
            raise ValueError(
                f"The model failed to produce a valid narrative for {cause} after "
                "four attempts."
            )

    return generated


def _read_label_map(label_map_path: str | Path) -> dict[str, int]:
    """Load and validate the project's cause-to-label mapping."""
    with Path(label_map_path).open(encoding="utf-8") as label_file:
        label_data = json.load(label_file)
    label2id = label_data.get("label2id")
    if not isinstance(label2id, dict) or not label2id:
        raise ValueError(f"{label_map_path} must contain a non-empty label2id mapping.")
    try:
        normalized = {str(cause): int(label) for cause, label in label2id.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label_map_path} contains invalid label IDs.") from error
    if len(normalized) != 8 or set(normalized.values()) != set(range(8)):
        raise ValueError(f"{label_map_path} must map exactly eight causes to IDs 0 through 7.")
    return normalized


def _read_training_data(train_path: Path, label2id: dict[str, int]) -> pd.DataFrame:
    """Load Experiment 1 train data and confirm its labels match label_map."""
    if not train_path.is_file():
        raise FileNotFoundError(f"Training CSV not found: {train_path}")
    train = pd.read_csv(train_path)
    required = {"narrative_clean", "CAUSE", "label"}
    if not required.issubset(train.columns):
        raise ValueError(f"{train_path} must contain {sorted(required)}.")
    if train["narrative_clean"].isna().any() or train["CAUSE"].isna().any():
        raise ValueError(f"{train_path} contains missing narratives or causes.")

    train["CAUSE"] = train["CAUSE"].astype(str)
    train["label"] = pd.to_numeric(train["label"], errors="raise").astype(int)
    expected_labels = train["CAUSE"].map(label2id)
    if expected_labels.isna().any() or not expected_labels.eq(train["label"]).all():
        raise ValueError(f"{train_path} causes and labels do not match label_map.")
    return train


def _load_api_key() -> str:
    """Load the Gemini key from the process environment or root .env file."""
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to the root .env file or set it in "
            "your shell; do not put the key in source files."
        )
    return api_key


class _GroqModel:
    """Adapter giving a Groq chat model the generate_content(prompt) interface."""

    def __init__(self, api_key: str, model_name: str) -> None:
        try:
            from groq import Groq
        except ImportError as error:
            raise ImportError("Install Groq support: pip install groq") from error
        self._client = Groq(api_key=api_key)
        self._model_name = model_name

    def generate_content(self, prompt: str, request_options: dict[str, Any] | None = None) -> Any:
        timeout = (request_options or {}).get("timeout", 120)
        response = self._client.chat.completions.create(
            model=self._model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            timeout=timeout,
        )
        return type("Response", (), {"text": response.choices[0].message.content or ""})()


def _build_model(provider: str, groq_model: str = GROQ_MODEL) -> Any:
    """Create the text-generation client for the chosen provider."""
    if provider == "groq":
        load_dotenv(ROOT / ".env")
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Add it to the root .env file or set it "
                "in your shell; do not put the key in source files."
            )
        return _GroqModel(api_key, groq_model)
    try:
        import google.generativeai as genai
    except ImportError as error:
        raise ImportError(
            "Install the project requirements to use Gemini: "
            "pip install -r requirements.txt"
        ) from error
    genai.configure(api_key=_load_api_key())
    return genai.GenerativeModel(GENERATION_MODEL)


def build_augmented_train(
    real_train_path: str | Path,
    synthetic_path: str | Path,
    output_path: str | Path,
    label_map_path: str | Path,
) -> pd.DataFrame:
    """Merge validated synthetic rows with Experiment 1 training data only."""
    label2id = _read_label_map(label_map_path)
    train_path = Path(real_train_path)
    synthetic_file = Path(synthetic_path)
    output_file = Path(output_path)
    train = _read_training_data(train_path, label2id)
    if not synthetic_file.is_file():
        raise FileNotFoundError(f"Synthetic narratives CSV not found: {synthetic_file}")

    synthetic = pd.read_csv(synthetic_file)
    if not set(SYNTHETIC_COLUMNS).issubset(synthetic.columns):
        raise ValueError(f"{synthetic_file} must contain {SYNTHETIC_COLUMNS}.")
    if synthetic.empty:
        raise ValueError(f"{synthetic_file} contains no synthetic narratives.")
    if not synthetic["source"].eq("synthetic").all():
        raise ValueError("Every generated row must have source='synthetic'.")
    if synthetic["narrative_clean"].isna().any() or synthetic["CAUSE"].isna().any():
        raise ValueError("Synthetic rows contain missing narratives or causes.")

    synthetic["narrative_clean"] = synthetic["narrative_clean"].astype(str)
    synthetic["CAUSE"] = synthetic["CAUSE"].astype(str)
    synthetic["label"] = pd.to_numeric(synthetic["label"], errors="raise").astype(int)
    expected_labels = synthetic["CAUSE"].map(label2id)
    if expected_labels.isna().any() or not expected_labels.eq(synthetic["label"]).all():
        raise ValueError("Every synthetic CAUSE must match its integer label in label_map.")
    if not synthetic["narrative_clean"].eq(
        synthetic["narrative_clean"].str.strip()
    ).all():
        raise ValueError("Synthetic narratives must not have leading or trailing whitespace.")
    if not synthetic["narrative_clean"].eq(
        synthetic["narrative_clean"].str.lower()
    ).all():
        raise ValueError("All synthetic narratives must be lowercase.")
    if synthetic["narrative_clean"].str.split().str.len().lt(50).any():
        raise ValueError("Every synthetic narrative must contain at least 50 words.")
    if any(
        cause.lower() in narrative.lower()
        for cause, narrative in zip(synthetic["CAUSE"], synthetic["narrative_clean"])
    ):
        raise ValueError("Synthetic narratives must not contain their cause label.")

    real_counts = train["CAUSE"].value_counts()
    rare_causes = set(real_counts[real_counts < TARGET_PER_CLASS].index)
    if not set(synthetic["CAUSE"]).issubset(rare_causes):
        raise ValueError("Synthetic narratives may be created only for rare training classes.")

    real_text = set(train["narrative_clean"].astype(str).str.strip().str.lower())
    synthetic_text = synthetic["narrative_clean"].str.strip().str.lower()
    if synthetic_text.isin(real_text).any():
        raise ValueError("A synthetic narrative is an exact copy of a real training narrative.")
    if synthetic_text.duplicated().any():
        raise ValueError("Synthetic narratives must be unique.")

    validation = pd.read_csv(VALIDATION_CSV)
    test = pd.read_csv(TEST_CSV)
    protected_text = set(validation["narrative_clean"].astype(str).str.strip())
    protected_text.update(test["narrative_clean"].astype(str).str.strip())

    real_rows = train.copy()
    real_rows["source"] = "real"
    output_columns = [*train.columns, "source"]
    synthetic_rows = pd.DataFrame(index=synthetic.index, columns=output_columns)
    for column in ("narrative_clean", "CAUSE", "label", "source"):
        synthetic_rows[column] = synthetic[column].to_numpy()
    if "NARRATIVE" in synthetic_rows.columns:
        synthetic_rows["NARRATIVE"] = synthetic["narrative_clean"].to_numpy()

    augmented = pd.concat(
        [real_rows[output_columns], synthetic_rows[output_columns]],
        ignore_index=True,
    )
    train_text = augmented["narrative_clean"].astype(str).str.strip()
    class_counts = augmented["CAUSE"].value_counts()
    checks = [
        ("all 8 classes are present", set(class_counts.index) == set(label2id)),
        (
            "all classes have at least 300 rows",
            all(int(class_counts.get(cause, 0)) >= TARGET_PER_CLASS for cause in label2id),
        ),
        ("no validation or test narrative appears in augmented train", train_text.isin(protected_text).sum() == 0),
        ("all synthetic narratives are lowercase", synthetic["narrative_clean"].eq(synthetic["narrative_clean"].str.lower()).all()),
        ("no synthetic narrative is an exact copy of a real one", not synthetic_text.isin(real_text).any()),
        ("total rows equal real plus synthetic rows", len(augmented) == len(train) + len(synthetic)),
    ]
    for description, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}: {description}")
    failed = [description for description, passed in checks if not passed]
    if failed:
        raise ValueError(f"Augmented training checks failed: {', '.join(failed)}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output_file, index=False)
    print("\nFinal class distribution:")
    print(class_counts.reindex(label2id).to_string())
    print(f"Total rows: {len(augmented)} ({len(train)} real + {len(synthetic)} synthetic)")
    return augmented


def _generate_csv(
    train_path: Path,
    label_map_path: Path,
    synthetic_path: Path,
    provider: str = DEFAULT_PROVIDER,
    groq_model: str = GROQ_MODEL,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Generate exactly enough validated text to reach the per-class target."""
    label2id = _read_label_map(label_map_path)
    train = _read_training_data(train_path, label2id)
    model = _build_model(provider, groq_model)
    counts = train["CAUSE"].value_counts()
    stats = {"failed_validation_attempts": 0, "regenerated_narratives": 0}
    generated_rows: list[dict[str, Any]] = []

    # Resume: keep rows already saved by an earlier (interrupted) run.
    synthetic_path.parent.mkdir(parents=True, exist_ok=True)
    if synthetic_path.is_file() and synthetic_path.stat().st_size > 0:
        generated_rows = pd.read_csv(synthetic_path).to_dict("records")
        print(f"Resuming with {len(generated_rows)} saved synthetic narratives.")
    else:
        pd.DataFrame(columns=SYNTHETIC_COLUMNS).to_csv(synthetic_path, index=False)

    for cause in label2id:
        count = int(counts.get(cause, 0))
        existing = [r["narrative_clean"] for r in generated_rows if r["CAUSE"] == cause]
        needed = max(0, TARGET_PER_CLASS - count - len(existing))
        if not needed:
            continue

        def save_row(narrative: str, cause: str = cause) -> None:
            row = {
                "narrative_clean": narrative,
                "CAUSE": cause,
                "label": label2id[cause],
                "source": "synthetic",
            }
            generated_rows.append(row)
            pd.DataFrame([row], columns=SYNTHETIC_COLUMNS).to_csv(
                synthetic_path, mode="a", header=False, index=False
            )
        training_examples = train.loc[train["CAUSE"].eq(cause), "narrative_clean"].astype(str).tolist()
        if len(training_examples) < 3:
            raise ValueError(f"Not enough real training narratives to prompt class {cause}.")

        class_stats: dict[str, int] = {}
        narratives = generate_synthetic_narratives(
            cause,
            needed,
            training_examples,
            model,
            stats=class_stats,
            existing=existing,
            on_narrative=save_row,
        )
        stats["failed_validation_attempts"] += class_stats["failed_validation_attempts"]
        stats["regenerated_narratives"] += class_stats["regenerated_narratives"]
        print(f"Generated {len(narratives)} narratives for {cause}.")
        used = groq_model if provider == "groq" else GENERATION_MODEL
        with (synthetic_path.parent / "synthetic_generation_log.csv").open("a", encoding="utf-8") as log:
            log.write(f"{used},{cause},{len(narratives)}\n")

    generated = pd.DataFrame(generated_rows, columns=SYNTHETIC_COLUMNS)
    return generated, stats


def main() -> None:
    """Generate synthetic data and build the Experiment 2 training CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=ROOT / "data" / "experiment1" / "train.csv",
    )
    parser.add_argument(
        "--label-map",
        type=Path,
        default=ROOT / "data" / "processed" / "label_map.json",
    )
    parser.add_argument(
        "--synthetic-csv",
        type=Path,
        default=ROOT / "data" / "processed" / "synthetic_narratives.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "experiment2" / "train.csv",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "gemini"],
        default=DEFAULT_PROVIDER,
        help="Text-generation API to use (default: %(default)s).",
    )
    parser.add_argument("--groq-model", default=GROQ_MODEL, help="Groq model id (default: %(default)s).")
    args = parser.parse_args()

    generated, stats = _generate_csv(
        args.train_csv, args.label_map, args.synthetic_csv, args.provider, args.groq_model
    )
    print(f"Total synthetic narratives generated: {len(generated)}")
    print(f"Invalid responses rejected: {stats['failed_validation_attempts']}")
    print(f"Narratives regenerated after invalid responses: {stats['regenerated_narratives']}")
    build_augmented_train(args.train_csv, args.synthetic_csv, args.output, args.label_map)


if __name__ == "__main__":
    main()
