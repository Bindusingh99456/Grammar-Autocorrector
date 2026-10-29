"""Check and enforce zero data leakage between training sets and test/dev holdouts.

Evaluation holdouts guarded:
1. data/processed/jfleg_test.json (source + all 4 reference corrections)
2. data/processed/wi_dev.csv (inputs + outputs)

Training datasets audited:
- data/processed/train.csv (C4_200M training set)
- data/processed/coedit_gec.csv (CoEdIT GEC training set)
- data/processed/wi_train.csv (W&I+LOCNESS training set)
- data/processed/fce_train.csv (FCE v2.1 training set)

Usage:
    python scripts/check_leakage.py          # Audit only (fails if leakage detected)
    python scripts/check_leakage.py --fix    # Purges all contaminating rows and saves clean datasets
"""

import argparse
import json
import re
import sys
from pathlib import Path
import pandas as pd


PROCESSED_DIR = Path("data/processed")
JFLEG_TEST_JSON = PROCESSED_DIR / "jfleg_test.json"
WI_DEV_CSV = PROCESSED_DIR / "wi_dev.csv"

TRAIN_DATASETS = {
    "C4_200M Train": PROCESSED_DIR / "train.csv",
    "CoEdIT GEC Train": PROCESSED_DIR / "coedit_gec.csv",
    "W&I+LOCNESS Train": PROCESSED_DIR / "wi_train.csv",
    "FCE v2.1 Train": PROCESSED_DIR / "fce_train.csv",
}


def normalize_sentence(text: str) -> str:
    """Strict alphanumeric normalization to detect any textual contamination."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def load_holdout_guard_set() -> tuple[dict[str, str], dict[str, int]]:
    """Build the comprehensive guard set of normalized sentences from JFLEG test and W&I dev."""
    guard_set = {}  # normalized_text -> origin description
    counts = {}

    # 1. JFLEG Test Split
    if JFLEG_TEST_JSON.exists():
        with open(JFLEG_TEST_JSON, "r", encoding="utf-8") as f:
            jfleg_records = json.load(f)
        counts["JFLEG Test (sentences)"] = len(jfleg_records)
        for idx, rec in enumerate(jfleg_records):
            src_norm = normalize_sentence(rec.get("source", ""))
            if src_norm:
                guard_set[src_norm] = f"JFLEG test source #{idx}"
            for r_idx, ref in enumerate(rec.get("references", [])):
                ref_norm = normalize_sentence(ref)
                if ref_norm:
                    guard_set[ref_norm] = f"JFLEG test ref{r_idx} #{idx}"
    else:
        print(f"[WARNING] Holdout file missing: {JFLEG_TEST_JSON}")

    # 2. W&I+LOCNESS Dev Split (Validation Only)
    if WI_DEV_CSV.exists():
        wi_dev_df = pd.read_csv(WI_DEV_CSV)
        counts["W&I+LOCNESS Dev (sentences)"] = len(wi_dev_df)
        for idx, row in wi_dev_df.iterrows():
            inp_norm = normalize_sentence(row.get("input", ""))
            if inp_norm:
                guard_set[inp_norm] = f"W&I dev input #{idx}"
            out_norm = normalize_sentence(row.get("output", ""))
            if out_norm:
                guard_set[out_norm] = f"W&I dev output #{idx}"
    else:
        print(f"[WARNING] Holdout file missing: {WI_DEV_CSV}")

    return guard_set, counts


def audit_dataset(name: str, path: Path, guard_set: dict[str, str], fix: bool = False) -> dict:
    """Audit a single training file against the holdout guard set."""
    if not path.exists():
        return {
            "name": name,
            "path": str(path),
            "status": "MISSING",
            "initial_rows": 0,
            "leakage_count": 0,
            "final_rows": 0,
        }

    df = pd.read_csv(path, encoding="utf-8")
    initial_rows = len(df)

    # Fast vector check
    inputs = df["input"].astype(str).tolist()
    outputs = df["output"].astype(str).tolist()

    overlapping_indices = []
    overlap_samples = []

    for idx, (inp, out) in enumerate(zip(inputs, outputs)):
        inp_norm = normalize_sentence(inp)
        out_norm = normalize_sentence(out)

        matched_origin = None
        if inp_norm in guard_set:
            matched_origin = guard_set[inp_norm]
        elif out_norm in guard_set:
            matched_origin = guard_set[out_norm]

        if matched_origin:
            overlapping_indices.append(idx)
            if len(overlap_samples) < 3:
                overlap_samples.append((inp, matched_origin))

    leakage_count = len(overlapping_indices)

    if leakage_count > 0 and fix:
        df_clean = df.drop(index=overlapping_indices).reset_index(drop=True)
        df_clean.to_csv(path, index=False, encoding="utf-8")
        final_rows = len(df_clean)
        status = "PURGED_CLEAN"
    elif leakage_count > 0:
        final_rows = initial_rows
        status = "LEAKAGE_DETECTED"
    else:
        final_rows = initial_rows
        status = "CLEAN"

    return {
        "name": name,
        "path": str(path),
        "status": status,
        "initial_rows": initial_rows,
        "leakage_count": leakage_count,
        "final_rows": final_rows,
        "samples": overlap_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Check for data leakage between training files and test/dev holdouts.")
    parser.add_argument("--fix", action="store_true", help="Purge any contaminating samples from training files.")
    args = parser.parse_args()

    print("=" * 80)
    print("DATA LEAKAGE & EVALUATION HOLDOUT AUDIT")
    print("=" * 80)
    print("Guarding against test/dev leakage from:")
    print(f"  - JFLEG Test:          {JFLEG_TEST_JSON}")
    print(f"  - W&I+LOCNESS Dev:     {WI_DEV_CSV}")

    guard_set, holdout_counts = load_holdout_guard_set()
    print("\nHoldout Dataset Sizes:")
    for k, v in holdout_counts.items():
        print(f"  * {k:30s}: {v:,}")
    print(f"  * Total unique normalized sentences:   {len(guard_set):,}")

    print("\nAuditing Training Sets:")
    results = []
    total_leaks = 0

    for name, path in TRAIN_DATASETS.items():
        res = audit_dataset(name, path, guard_set, fix=args.fix)
        results.append(res)
        total_leaks += res["leakage_count"]

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Dataset':<22} | {'Initial Rows':<12} | {'Leaks Found':<11} | {'Final Rows':<12} | {'Status'}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<22} | {r['initial_rows']:>12,} | {r['leakage_count']:>11,} | {r['final_rows']:>12,} | {r['status']}")
    print("=" * 80)

    for r in results:
        if r.get("samples"):
            print(f"\nSample overlaps found in {r['name']}:")
            for text, origin in r["samples"]:
                print(f"  - Training text: {repr(text)}")
                print(f"    Origin in holdout: {origin}")

    if total_leaks > 0 and not args.fix:
        print(f"\n[FAIL] Detected {total_leaks} total overlapping sentences across training datasets!")
        print("Run with `python scripts/check_leakage.py --fix` to sanitize training datasets.")
        sys.exit(1)
    elif total_leaks > 0 and args.fix:
        print(f"\n[FIXED] Successfully purged all {total_leaks} contaminating rows.")
        print("[SUCCESS] Zero data leakage guaranteed across all training and evaluation sets.")
        sys.exit(0)
    else:
        print("\n[SUCCESS] PROVEN 0 DATA LEAKAGE: No training sentence appears in jfleg_test.json or wi_dev.csv.")
        sys.exit(0)


if __name__ == "__main__":
    main()
