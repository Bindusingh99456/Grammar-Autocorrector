"""Prepare BEA-2019 W&I+LOCNESS and FCE v2.1 datasets.

Requirements from Phase M1:
- W&I+LOCNESS: download https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz
  extract, apply each annotation's edits to build corrected text, split into sentence pairs (max 64 words).
  Train split -> data/processed/wi_train.csv
  Dev split -> data/processed/wi_dev.csv (validation only, never training)
- FCE v2.1: try the BEA-2019 page link for fce_v2.1.bea19.tar.gz and process it the same way into fce_train.csv.
  If the link 404s, say so and continue without it.
"""

import csv
import json
import re
import sys
import tarfile
import urllib.request
import urllib.error
from pathlib import Path
from nltk.tokenize import PunktSentenceTokenizer

RAW_DIR = Path("data/raw/bea2019")
PROCESSED_DIR = Path("data/processed")

WI_TAR_URL = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"
WI_TAR_FILE = RAW_DIR / "wi+locness_v2.1.bea19.tar.gz"

FCE_TAR_URL = "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz"
FCE_TAR_FILE = RAW_DIR / "fce_v2.1.bea19.tar.gz"

WI_TRAIN_CSV = PROCESSED_DIR / "wi_train.csv"
WI_DEV_CSV = PROCESSED_DIR / "wi_dev.csv"
FCE_TRAIN_CSV = PROCESSED_DIR / "fce_train.csv"

MAX_WORDS = 64
tokenizer = PunktSentenceTokenizer()


def download_file(url: str, dest_path: Path) -> bool:
    """Download a file from url with HTTP status checking."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"Archive already present at {dest_path} ({dest_path.stat().st_size:,} bytes).")
        return True

    print(f"Downloading from {url} to {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
            if resp.status == 404:
                print(f"[404 NOT FOUND] The link {url} returned 404.")
                return False
            # Stream download
            chunk_size = 64 * 1024
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
        print(f"Downloaded successfully: {dest_path} ({dest_path.stat().st_size:,} bytes).")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"[404 NOT FOUND] The link {url} returned 404.")
        else:
            print(f"[HTTP ERROR {e.code}] Failed downloading {url}: {e.reason}")
        if dest_path.exists():
            dest_path.unlink()
        return False
    except Exception as e:
        print(f"[DOWNLOAD ERROR] Failed downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def extract_sentence_pairs_from_text(text: str, edits: list) -> list:
    """Extract aligned sentence pairs (input, output) applying character edits per sentence."""
    pairs = []
    # Match non-newline sequences (paragraphs)
    for p_match in re.finditer(r"[^\n]+", text):
        p_start = p_match.start()
        p_end = p_match.end()
        para = p_match.group()

        # Edits that fall inside this paragraph
        p_edits = []
        for e in edits:
            # Edit structure: [start, end, replacement, ...]
            e_start, e_end, e_cor = e[0], e[1], e[2]
            if e_start >= p_start and e_end <= p_end:
                p_edits.append([e_start - p_start, e_end - p_start, e_cor])

        # Sentence-tokenize paragraph
        s_spans = list(tokenizer.span_tokenize(para))
        for s_start, s_end in s_spans:
            orig_s = para[s_start:s_end].strip()
            if not orig_s:
                continue

            # Edits strictly inside sentence span
            s_edits = [e for e in p_edits if e[0] >= s_start and e[1] <= s_end]
            # Sort edits descending by start offset
            s_edits_sorted = sorted(s_edits, key=lambda x: (x[0], x[1]), reverse=True)

            cor_s = para[s_start:s_end]
            for es, ee, ec in s_edits_sorted:
                if ec is None:
                    # Detection only, original unchanged
                    continue
                rel_s = es - s_start
                rel_e = ee - s_start
                cor_s = cor_s[:rel_s] + ec + cor_s[rel_e:]

            # Normalize whitespaces
            orig_s = re.sub(r"\s+", " ", orig_s).strip()
            cor_s = re.sub(r"\s+", " ", cor_s).strip()

            if not orig_s or not cor_s:
                continue

            # Check length limit (<= 64 words)
            if len(orig_s.split()) <= MAX_WORDS and len(cor_s.split()) <= MAX_WORDS:
                pairs.append((orig_s, cor_s))

    return pairs


def process_tar_json_files(tar_path: Path, member_files: list, output_csv: Path, description: str):
    """Process a list of JSON member files inside a tarball into a CSV file."""
    print(f"\nProcessing {description} from {tar_path.name}...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for m_name in member_files:
            try:
                f = tar.extractfile(m_name)
                if f is None:
                    print(f"  [WARNING] Could not read {m_name}")
                    continue
                file_pairs = []
                for line in f:
                    item = json.loads(line.decode("utf-8"))
                    text = item.get("text", "")
                    for ann_id, edits in item.get("edits", []):
                        pairs = extract_sentence_pairs_from_text(text, edits)
                        file_pairs.extend(pairs)
                print(f"  Parsed {m_name}: {len(file_pairs):,} sentence pairs.")
                all_pairs.extend(file_pairs)
            except KeyError:
                print(f"  [WARNING] Member {m_name} not found in archive.")

    # Deduplicate exact duplicate sentence pairs
    deduped = []
    seen = set()
    for inp, out in all_pairs:
        key = (inp, out)
        if key not in seen:
            seen.add(key)
            deduped.append({"input": inp, "output": out})

    changed_count = sum(1 for r in deduped if r["input"] != r["output"])
    print(f"Total extracted for {description}: {len(deduped):,} pairs ({changed_count:,} with edits, {len(deduped) - changed_count:,} unchanged).")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "output"])
        writer.writeheader()
        writer.writerows(deduped)

    print(f"Saved {description} -> {output_csv} ({len(deduped):,} rows, {output_csv.stat().st_size:,} bytes)")
    if deduped:
        print("Sample pairs (first 2):")
        for i, r in enumerate(deduped[:2], 1):
            print(f"  [{i}] Input:  {r['input']}")
            print(f"      Output: {r['output']}")
    return len(deduped)


def main():
    print("=" * 70)
    print("BEA-2019 DATASET PREPARATION (W&I+LOCNESS & FCE v2.1)")
    print("=" * 70)

    # 1. W&I+LOCNESS
    print("\n--- Step 1: W&I+LOCNESS v2.1 ---")
    wi_ok = download_file(WI_TAR_URL, WI_TAR_FILE)
    if not wi_ok:
        print("[ERROR] Could not download W&I+LOCNESS archive.", file=sys.stderr)
        sys.exit(1)

    wi_train_members = [
        "wi+locness/json/A.train.json",
        "wi+locness/json/B.train.json",
        "wi+locness/json/C.train.json",
    ]
    wi_dev_members = [
        "wi+locness/json/A.dev.json",
        "wi+locness/json/B.dev.json",
        "wi+locness/json/C.dev.json",
        "wi+locness/json/N.dev.json",
    ]

    process_tar_json_files(WI_TAR_FILE, wi_train_members, WI_TRAIN_CSV, "W&I+LOCNESS Train Split")
    process_tar_json_files(WI_TAR_FILE, wi_dev_members, WI_DEV_CSV, "W&I+LOCNESS Dev Split (Validation Only)")

    # 2. FCE v2.1
    print("\n--- Step 2: FCE v2.1 ---")
    fce_ok = download_file(FCE_TAR_URL, FCE_TAR_FILE)
    if fce_ok:
        fce_train_members = ["fce/json/fce.train.json"]
        process_tar_json_files(FCE_TAR_FILE, fce_train_members, FCE_TRAIN_CSV, "FCE v2.1 Train Split")
    else:
        print("[NOTICE] FCE v2.1 download link 404'd or failed. Continuing without FCE as permitted by instructions.")

    print("\n[COMPLETE] W&I+LOCNESS and FCE dataset preparation finished.")


if __name__ == "__main__":
    main()
