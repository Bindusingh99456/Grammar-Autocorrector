import json
import random
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


C4_INPUT_FILE = Path("data/raw/c4_200m/subset.csv")
PROCESSED_DIR = Path("data/processed")
TRAIN_FILE = PROCESSED_DIR / "train.csv"
VAL_FILE = PROCESSED_DIR / "val.csv"

JFLEG_RAW_DIR = Path("data/raw/jfleg")
JFLEG_DEV_JSON = PROCESSED_DIR / "jfleg_dev.json"
JFLEG_TEST_JSON = PROCESSED_DIR / "jfleg_test.json"


def clean_c4_data():
    print(f"Reading raw C4 subset from {C4_INPUT_FILE}...")
    df = pd.read_csv(C4_INPUT_FILE, encoding="utf-8")
    initial_count = len(df)
    print(f"Initial raw rows: {initial_count:,}")

    # Drop nulls and strip whitespace
    df = df.dropna(subset=["input", "output"])
    df["input"] = df["input"].astype(str).str.strip()
    df["output"] = df["output"].astype(str).str.strip()

    # Drop empty strings
    df = df[(df["input"] != "") & (df["output"] != "")]

    # Drop identical pairs
    df = df[df["input"] != df["output"]]

    # Drop pairs longer than 64 words
    input_word_counts = df["input"].str.split().str.len()
    output_word_counts = df["output"].str.split().str.len()
    df = df[(input_word_counts <= 64) & (output_word_counts <= 64)]

    cleaned_count = len(df)
    print(f"Cleaned rows (non-empty, non-identical, <= 64 words): {cleaned_count:,} (dropped {initial_count - cleaned_count:,})")

    # Split 95 / 5
    print("Splitting 95% train / 5% validation (random_state=42)...")
    train_df, val_df = train_test_split(df, test_size=0.05, random_state=42, shuffle=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_FILE, index=False, encoding="utf-8")
    val_df.to_csv(VAL_FILE, index=False, encoding="utf-8")

    print(f"Saved train set: {TRAIN_FILE} ({len(train_df):,} rows)")
    print(f"Saved validation set: {VAL_FILE} ({len(val_df):,} rows)")

    return train_df, val_df


def process_jfleg_split(split_name: str, output_path: Path):
    split_dir = JFLEG_RAW_DIR / split_name
    src_file = split_dir / f"{split_name}.src"
    ref_files = [split_dir / f"{split_name}.ref{i}" for i in range(4)]

    for path in [src_file] + ref_files:
        if not path.exists():
            raise FileNotFoundError(f"Missing JFLEG file: {path}")

    with open(src_file, "r", encoding="utf-8") as f:
        src_lines = [line.strip() for line in f]

    refs_content = []
    for r_file in ref_files:
        with open(r_file, "r", encoding="utf-8") as f:
            refs_content.append([line.strip() for line in f])

    total_lines = len(src_lines)
    for i, ref_list in enumerate(refs_content):
        if len(ref_list) != total_lines:
            raise ValueError(f"Line count mismatch in {split_name}.ref{i}: {len(ref_list)} vs {total_lines}")

    records = []
    for idx in range(total_lines):
        record = {
            "source": src_lines[idx],
            "references": [refs_content[r_idx][idx] for r_idx in range(4)],
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Converted JFLEG {split_name} -> {output_path} ({len(records):,} sentences, 4 references each)")
    return records


def main():
    print("=== Step 1: Processing C4 200M Subset ===")
    train_df, val_df = clean_c4_data()

    print("\n=== Step 2: Processing JFLEG Dev and Test Sets ===")
    jfleg_dev = process_jfleg_split("dev", JFLEG_DEV_JSON)
    jfleg_test = process_jfleg_split("test", JFLEG_TEST_JSON)

    print("\n=== Dataset Summary ===")
    print(f"Train split: {len(train_df):,} rows")
    print(f"Validation split: {len(val_df):,} rows")
    print(f"JFLEG Dev: {len(jfleg_dev):,} sentences")
    print(f"JFLEG Test: {len(jfleg_test):,} sentences")

    print("\n=== 5 Random Examples: Training Data (C4) ===")
    sample_indices = random.sample(range(len(train_df)), min(5, len(train_df)))
    for rank, idx in enumerate(sample_indices, 1):
        row = train_df.iloc[idx]
        print(f"[{rank}] Input:  {row['input']}")
        print(f"    Output: {row['output']}\n")

    print("=== 5 Random Examples: JFLEG Dev Data ===")
    dev_sample = random.sample(jfleg_dev, min(5, len(jfleg_dev)))
    for rank, ex in enumerate(dev_sample, 1):
        print(f"[{rank}] Source: {ex['source']}")
        print(f"    Ref 0:  {ex['references'][0]}")
        print(f"    Ref 1:  {ex['references'][1]}\n")


if __name__ == "__main__":
    random.seed(42)
    main()
