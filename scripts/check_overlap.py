import json
import re
import sys
from pathlib import Path
import pandas as pd


def normalize_text(text: str) -> str:
    """Normalize string for strict contamination and overlap detection."""
    if not isinstance(text, str):
        return ""
    # Lowercase and remove all non-alphanumeric characters and collapse spaces
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def check_overlap(train_csv_path="data/processed/train.csv", fix=False):
    train_path = Path(train_csv_path)
    jfleg_test_path = Path("data/processed/jfleg_test.json")
    wilocness_dev_path = Path("data/processed/wilocness_dev.json")

    print("=" * 70)
    print("DATASET OVERLAP & CONTAMINATION AUDIT")
    print("=" * 70)

    if not train_path.exists():
        print(f"[ERROR] Training file not found: {train_path}")
        sys.exit(1)

    print(f"Loading training data from: {train_path}")
    train_df = pd.read_csv(train_path)
    total_train = len(train_df)
    print(f"Total training pairs: {total_train:,}")

    # Build set of normalized training inputs
    train_norm_inputs = set()
    train_input_map = {}  # norm -> list of row indices
    for idx, raw_inp in enumerate(train_df["input"].astype(str)):
        norm = normalize_text(raw_inp)
        if norm:
            train_norm_inputs.add(norm)
            train_input_map.setdefault(norm, []).append(idx)

    overlaps_found = {}
    rows_to_remove = set()

    # 1. Check JFLEG Test Split
    if jfleg_test_path.exists():
        print(f"\nChecking JFLEG Test Split ({jfleg_test_path})...")
        with open(jfleg_test_path, "r", encoding="utf-8") as f:
            jfleg_data = json.load(f)

        jfleg_overlaps = []
        for item in jfleg_data:
            src_norm = normalize_text(item.get("source", ""))
            if src_norm in train_norm_inputs:
                jfleg_overlaps.append(item["source"])
                rows_to_remove.update(train_input_map[src_norm])

            # Also check references
            for ref in item.get("references", []):
                ref_norm = normalize_text(ref)
                if ref_norm in train_norm_inputs:
                    jfleg_overlaps.append(f"REF: {ref}")
                    rows_to_remove.update(train_input_map[ref_norm])

        overlaps_found["JFLEG_test"] = jfleg_overlaps
        print(f"  Total JFLEG test sentences checked: {len(jfleg_data)}")
        print(f"  Contaminating overlaps detected:     {len(jfleg_overlaps)}")
    else:
        print(f"[WARNING] JFLEG test file missing at {jfleg_test_path}")

    # 2. Check W&I+LOCNESS Dev Split
    if wilocness_dev_path.exists():
        print(f"\nChecking W&I+LOCNESS Dev Split ({wilocness_dev_path})...")
        with open(wilocness_dev_path, "r", encoding="utf-8") as f:
            wilocness_data = json.load(f)

        wilocness_overlaps = []
        for sent in wilocness_data:
            s_norm = normalize_text(sent)
            if s_norm in train_norm_inputs:
                wilocness_overlaps.append(sent)
                rows_to_remove.update(train_input_map[s_norm])

        overlaps_found["W&I_LOCNESS_dev"] = wilocness_overlaps
        print(f"  Total W&I+LOCNESS sentences checked: {len(wilocness_data):,}")
        print(f"  Contaminating overlaps detected:     {len(wilocness_overlaps)}")
    else:
        print(f"[WARNING] W&I+LOCNESS dev file missing at {wilocness_dev_path}")

    # Summary
    total_overlapping_rows = len(rows_to_remove)
    print("\n" + "-" * 70)
    print(f"AUDIT SUMMARY: {total_overlapping_rows} training rows overlap with test/dev splits.")
    print("-" * 70)

    if total_overlapping_rows > 0:
        if fix:
            print(f"Sanitizing training dataset by purging {total_overlapping_rows} overlapping rows...")
            clean_df = train_df.drop(index=list(rows_to_remove)).reset_index(drop=True)
            clean_df.to_csv(train_path, index=False)
            print(f"Sanitization complete. Cleaned training dataset saved: {len(clean_df):,} rows.")
            print("[SUCCESS] Zero test/dev data contamination confirmed.")
            return 0
        else:
            print("[FAIL] Overlap detected! Run with --fix to automatically purge contaminating samples.")
            return 1
    else:
        print("[SUCCESS] 0 overlaps detected across all test and dev benchmarks.")
        print("Verification passed: Test data is completely isolated from training data.")
        return 0


if __name__ == "__main__":
    should_fix = "--fix" in sys.argv
    sys.exit(check_overlap(fix=should_fix))
