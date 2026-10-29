"""Prepare CoEdIT GEC dataset.

Requirements from Phase M1:
- load_dataset("grammarly/coedit")
- keep rows with task == "gec"
- split the "src" field into instruction and text
- save data/processed/coedit_gec.csv (input,output)
"""

import csv
import re
from pathlib import Path
from datasets import load_dataset


PROCESSED_DIR = Path("data/processed")
OUTPUT_FILE = PROCESSED_DIR / "coedit_gec.csv"
MAX_WORDS = 64


def prepare_coedit():
    print("=" * 60)
    print("Preparing CoEdIT GEC Dataset")
    print("=" * 60)

    print("Loading 'grammarly/coedit' dataset from Hugging Face Hub...")
    ds = load_dataset("grammarly/coedit")

    # We inspect all rows with task == 'gec' across available splits
    splits_to_check = ["train"]
    if "validation" in ds:
        # User specified CoEdIT as a training source; keep train split
        print(f"Dataset splits available: {list(ds.keys())}. Using 'train' split.")

    train_data = ds["train"]
    print(f"Total rows in CoEdIT train split: {len(train_data):,}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    dropped_empty = 0
    dropped_identical = 0
    dropped_long = 0

    for item in train_data:
        if item.get("task") != "gec":
            continue

        src = item.get("src", "")
        tgt = item.get("tgt", "")

        # Split on the first colon into instruction and sentence text
        if ":" in src:
            _, text = src.split(":", 1)
        else:
            text = src

        inp = re.sub(r"\s+", " ", text).strip()
        out = re.sub(r"\s+", " ", tgt).strip()

        if not inp or not out:
            dropped_empty += 1
            continue

        if inp == out:
            dropped_identical += 1
            continue

        inp_words = len(inp.split())
        out_words = len(out.split())
        if inp_words > MAX_WORDS or out_words > MAX_WORDS:
            dropped_long += 1
            continue

        rows.append({"input": inp, "output": out})

    print(f"Extracted GEC pairs: {len(rows):,}")
    print(f"Dropped: empty={dropped_empty}, identical={dropped_identical}, >{MAX_WORDS} words={dropped_long}")

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Successfully saved to {OUTPUT_FILE} ({len(rows):,} pairs, {OUTPUT_FILE.stat().st_size:,} bytes)")
    print("\nSample pairs (first 3):")
    for idx, r in enumerate(rows[:3], 1):
        print(f"[{idx}] Input:  {r['input']}")
        print(f"    Output: {r['output']}\n")

    return len(rows)


if __name__ == "__main__":
    prepare_coedit()
