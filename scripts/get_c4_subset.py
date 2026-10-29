import csv
import sys
from pathlib import Path


RAW_DIR = Path("data/raw/c4_200m")
OUTPUT_FILE = RAW_DIR / "subset.csv"
TARGET_ROWS = 400_000


def download_with_streaming():
    from datasets import load_dataset

    print("Attempting to stream dataset 'liweili/c4_200m' via datasets library...", flush=True)
    ds = load_dataset("liweili/c4_200m", split="train", streaming=True)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "output"])

        for row in ds:
            inp = row.get("input", "")
            out = row.get("output", "")
            writer.writerow([inp, out])
            count += 1

            if count % 25000 == 0:
                print(f"Streamed {count:,} / {TARGET_ROWS:,} rows...", flush=True)

            if count >= TARGET_ROWS:
                break

    print(f"Successfully collected {count:,} rows via datasets streaming.", flush=True)
    return True


def download_with_hub_fallback():
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    print("Falling back to huggingface_hub download for 'liweili/c4_200m'...", flush=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    
    # Check for converted parquet shards on Hugging Face Hub
    try:
        files = api.list_repo_files(repo_id="liweili/c4_200m", revision="refs/convert/parquet", repo_type="dataset")
        parquet_files = [f for f in files if f.endswith(".parquet")]
        revision = "refs/convert/parquet"
    except Exception:
        files = api.list_repo_files(repo_id="liweili/c4_200m", repo_type="dataset")
        parquet_files = [f for f in files if f.endswith(".parquet")]
        revision = None

    print(f"Discovered data files in repository (revision={revision}): {parquet_files[:5]}", flush=True)

    if not parquet_files:
        raise RuntimeError("No parquet data files found in Hugging Face repository.")

    with open(OUTPUT_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "output"])
        total_written = 0

        for filename in parquet_files:
            print(f"Fetching shard via huggingface_hub: {filename}...", flush=True)
            local_path = hf_hub_download(
                repo_id="liweili/c4_200m",
                filename=filename,
                revision=revision,
                repo_type="dataset",
            )
            print(f"Downloaded shard to {local_path}. Reading rows...", flush=True)

            parquet_file = pq.ParquetFile(local_path)
            for batch in parquet_file.iter_batches(batch_size=10000, columns=["input", "output"]):
                df_batch = batch.to_pandas()
                for _, r in df_batch.iterrows():
                    writer.writerow([r["input"], r["output"]])
                    total_written += 1

                    if total_written % 25000 == 0:
                        print(f"Extracted {total_written:,} / {TARGET_ROWS:,} rows...", flush=True)

                    if total_written >= TARGET_ROWS:
                        break
                if total_written >= TARGET_ROWS:
                    break

            if total_written >= TARGET_ROWS:
                break

    print(f"Successfully saved {total_written:,} rows to {OUTPUT_FILE} via huggingface_hub fallback.", flush=True)
    return True


def main():
    if OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0:
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_lines = sum(1 for _ in f) - 1
        if existing_lines >= TARGET_ROWS:
            print(f"Found existing dataset at {OUTPUT_FILE} ({existing_lines:,} rows). Skipping download.", flush=True)
            return
        else:
            print(f"Existing dataset has {existing_lines:,} rows, need {TARGET_ROWS:,}. Refreshing...", flush=True)

    method_used = None
    try:
        if download_with_streaming():
            method_used = "datasets streaming"
    except Exception as exc:
        print(f"Streaming failed with error: {exc}", flush=True)
        print("Switching to huggingface_hub fallback...", flush=True)
        try:
            if download_with_hub_fallback():
                method_used = "huggingface_hub fallback (refs/convert/parquet)"
        except Exception as fallback_exc:
            print(f"Fallback also failed: {fallback_exc}", file=sys.stderr, flush=True)
            sys.exit(1)

    print(f"Download completed successfully. Method that worked: {method_used}", flush=True)


if __name__ == "__main__":
    main()
