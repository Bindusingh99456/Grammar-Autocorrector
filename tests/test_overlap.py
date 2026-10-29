import json
import re
from pathlib import Path
import pandas as pd


def normalize_text(text: str) -> str:
    """Normalize string for strict contamination and overlap detection."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def test_zero_test_data_overlap():
    """Verify that test data (JFLEG test, W&I+LOCNESS dev) never appears in ANY training dataset."""
    train_files = [
        Path("data/processed/train.csv"),
        Path("data/processed/coedit_gec.csv"),
        Path("data/processed/wi_train.csv"),
        Path("data/processed/fce_train.csv"),
    ]
    jfleg_test_path = Path("data/processed/jfleg_test.json")
    wi_dev_path = Path("data/processed/wi_dev.csv")

    assert jfleg_test_path.exists(), "JFLEG test data must exist"
    assert wi_dev_path.exists(), "W&I+LOCNESS dev data must exist"

    # Build benchmark holdout sets
    with open(jfleg_test_path, "r", encoding="utf-8") as f:
        jfleg_data = json.load(f)

    jfleg_norms = set()
    for item in jfleg_data:
        s_norm = normalize_text(item.get("source", ""))
        if s_norm:
            jfleg_norms.add(s_norm)
        for ref in item.get("references", []):
            r_norm = normalize_text(ref)
            if r_norm:
                jfleg_norms.add(r_norm)

    wi_dev_df = pd.read_csv(wi_dev_path)
    wi_dev_norms = set()
    for s in wi_dev_df["input"].astype(str):
        n = normalize_text(s)
        if n:
            wi_dev_norms.add(n)
    for s in wi_dev_df["output"].astype(str):
        n = normalize_text(s)
        if n:
            wi_dev_norms.add(n)

    # Verify each training dataset
    for path in train_files:
        assert path.exists(), f"Training file {path} must exist"
        df = pd.read_csv(path)
        inputs = df["input"].astype(str).tolist()
        outputs = df["output"].astype(str).tolist()

        for idx, (inp, out) in enumerate(zip(inputs, outputs)):
            inp_norm = normalize_text(inp)
            out_norm = normalize_text(out)

            if inp_norm:
                assert inp_norm not in jfleg_norms, f"JFLEG test leakage in {path.name} at row {idx}: {inp}"
                assert inp_norm not in wi_dev_norms, f"W&I dev leakage in {path.name} at row {idx}: {inp}"
            if out_norm:
                assert out_norm not in jfleg_norms, f"JFLEG test leakage in {path.name} output at row {idx}: {out}"
                assert out_norm not in wi_dev_norms, f"W&I dev leakage in {path.name} output at row {idx}: {out}"
