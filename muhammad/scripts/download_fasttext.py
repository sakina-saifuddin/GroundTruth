"""Download the FastText Common Crawl binary model (cc.en.300.bin) via fasttext.util."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) > 1:
        dest_dir = Path(sys.argv[1])
    else:
        dest_dir = Path(__file__).resolve().parents[2] / "muhammad" / "fasttext"

    dest_dir.mkdir(parents=True, exist_ok=True)
    bin_path = dest_dir / "cc.en.300.bin"

    if bin_path.is_file():
        print(f"Already exists: {bin_path}")
        print("Nothing to do.")
        return

    try:
        import fasttext.util
    except ImportError:
        print("fasttext is not installed. Run:  pip install fasttext")
        sys.exit(1)

    import os
    # fasttext.util.download_model saves to the current directory
    original_cwd = os.getcwd()
    os.chdir(dest_dir)
    try:
        print(f"Downloading cc.en.300.bin (~4.2 GB) to {dest_dir} ...")
        fasttext.util.download_model("en", if_exists="ignore")
    finally:
        os.chdir(original_cwd)

    print(f"\nDone. Model at:\n  {bin_path}")
    print()
    print("Update fasttext_path in your experiment scripts to:")
    print(f'  "fasttext_path": "{bin_path}",')


if __name__ == "__main__":
    main()
