"""Analyze BERT WordPiece token lengths for the cleaned PHMSA narratives."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_prep import LABELS, REPORT_END, REPORT_START  # noqa: E402


def load_tokenizer():
    """Load the pretrained tokenizer or explain the missing dependency/network."""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "The token-length report requires transformers. Install it with "
            "`pip install -r requirements.txt`."
        ) from error

    try:
        return AutoTokenizer.from_pretrained("bert-base-uncased")
    except OSError as error:
        raise RuntimeError(
            "Could not load bert-base-uncased. Check your internet connection or "
            "whether the tokenizer is already cached locally."
        ) from error


def describe_lengths(lengths: pd.Series) -> dict[str, float | int]:
    """Calculate summary statistics and strict threshold exceedance counts."""
    return {
        "median": float(lengths.median()),
        "mean": float(lengths.mean()),
        "95th percentile": float(lengths.quantile(0.95)),
        "maximum": int(lengths.max()),
        "over 128": int(lengths.gt(128).sum()),
        "over 256": int(lengths.gt(256).sum()),
        "over 512": int(lengths.gt(512).sum()),
    }


def replace_token_report(report_path: Path, section: str) -> None:
    """Replace the marked token-length section in the generated data report."""
    if not report_path.is_file():
        raise FileNotFoundError(
            f"{report_path} does not exist. Run `python scripts/prepare_data.py` first."
        )
    report = report_path.read_text(encoding="utf-8")
    if REPORT_START not in report or REPORT_END not in report:
        raise ValueError(f"{report_path} is missing the generated token-report markers.")
    before, remainder = report.split(REPORT_START, 1)
    _, after = remainder.split(REPORT_END, 1)
    report_path.write_text(
        f"{before}{REPORT_START}\n{section.strip()}\n{REPORT_END}{after}",
        encoding="utf-8",
    )


def main() -> None:
    """Tokenize all prepared narratives, chart lengths, and update the report."""
    csv_path = ROOT / "data" / "processed" / "clean_full.csv"
    report_path = ROOT / "reports" / "data_report.md"
    figures_dir = ROOT / "reports" / "figures"
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"{csv_path} does not exist. Run `python scripts/prepare_data.py` first."
        )

    data = pd.read_csv(csv_path)
    tokenizer = load_tokenizer()
    encoded = tokenizer(
        data["narrative_clean"].tolist(),
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    lengths = pd.Series([len(token_ids) for token_ids in encoded["input_ids"]], index=data.index)
    data["bert_tokens"] = lengths

    overall = describe_lengths(lengths)
    rows = ["| Group | Median | Mean | 95th percentile | Max | >128 | >256 | >512 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, group in [("Overall", data), *[(cause, data.loc[data["CAUSE"].eq(cause)]) for cause in LABELS]]:
        stats = describe_lengths(group["bert_tokens"])
        rows.append(
            "| {} | {:.1f} | {:.1f} | {:.1f} | {} | {} | {} | {} |".format(
                name,
                stats["median"],
                stats["mean"],
                stats["95th percentile"],
                stats["maximum"],
                stats["over 128"],
                stats["over 256"],
                stats["over 512"],
            )
        )

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist(lengths, bins=40, color="#477a68", edgecolor="white")
    axis.axvline(256, color="#d99b42", linestyle="--", label="256 tokens")
    axis.axvline(512, color="#b6533c", linestyle="--", label="512 tokens")
    axis.set_title("BERT WordPiece token lengths")
    axis.set_xlabel("Tokens, including special tokens")
    axis.set_ylabel("Narratives")
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures_dir / "bert_token_lengths.png", dpi=160)
    plt.close(figure)

    example_index = None
    example_tokens = None
    for index, text in enumerate(data["narrative_clean"]):
        input_ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        if any(token.startswith("##") for token in tokens):
            example_index, example_tokens = index, tokens
            break
    if example_index is None:
        example_index = 0
        token_ids = tokenizer(
            data.loc[example_index, "narrative_clean"],
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        example_tokens = tokenizer.convert_ids_to_tokens(token_ids)

    example_text = data.loc[example_index, "narrative_clean"]
    example_preview = example_text[:240].replace("`", "'")
    section = (
        f"Token counts include `[CLS]` and `[SEP]`. No truncation was applied.\n\n"
        + "\n".join(rows)
        + f"\n\nExample narrative: `{example_preview}`\n\n"
        + f"WordPiece tokens: `{' '.join(example_tokens)}`\n"
    )
    replace_token_report(report_path, section)
    print(f"Overall token statistics: {overall}")
    print(f"Example WordPiece tokens: {' '.join(example_tokens)}")
    print(f"Updated {report_path} and {figures_dir / 'bert_token_lengths.png'}")


if __name__ == "__main__":
    main()
