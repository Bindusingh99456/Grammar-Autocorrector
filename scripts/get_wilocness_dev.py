import json
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
import nltk
from nltk.tokenize import sent_tokenize


def fetch_wilocness_dev():
    output_raw_dir = Path("data/raw/wilocness")
    output_raw_dir.mkdir(parents=True, exist_ok=True)
    output_raw_file = output_raw_dir / "wi_locness_dev.parquet"

    print("Fetching W&I+LOCNESS dev split from Hugging Face (martinsr/wi_locness)...")
    downloaded_path = hf_hub_download(
        repo_id="martinsr/wi_locness",
        filename="wi_locness_dev.parquet",
        repo_type="dataset",
    )

    # Copy / save to local data/raw/wilocness/
    df = pd.read_parquet(downloaded_path)
    df.to_parquet(output_raw_file)
    print(f"Saved raw parquet to {output_raw_file} ({len(df)} essays/paragraphs)")

    # Extract sentences for testing and overlap detection
    all_sentences = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue
        sents = sent_tokenize(text)
        for s in sents:
            s_clean = s.strip()
            if len(s_clean) > 3:
                all_sentences.append(s_clean)

    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_json = processed_dir / "wilocness_dev.json"

    with open(processed_json, "w", encoding="utf-8") as f:
        json.dump(all_sentences, f, indent=2)

    print(f"Saved {len(all_sentences):,} segmented sentences to {processed_json}")
    return processed_json


if __name__ == "__main__":
    fetch_wilocness_dev()
