"""
Sample 500 already-correct C4 sentences for False-Positive Rate (FPR) measurement.
Sentences are extracted from clean validation targets, filtered for length,
natural syntax, and lack of noise/artifacts.
"""

import csv
import json
import re
from pathlib import Path


def is_clean_sentence(s: str) -> bool:
    s = s.strip()
    if len(s) < 25 or len(s) > 120:
        return False
    # Must end with normal sentence punctuation
    if not re.search(r"[.!?]$", s):
        return False
    # Must start with capital letter
    if not s[0].isupper():
        return False
    # No weird markup or repetitive symbols
    if re.search(r"http|www|[<>{}\[\]_~^\\/|#*]|([a-zA-Z])\1{3,}", s):
        return False
    # No unclosed quotes
    if s.count('"') % 2 != 0:
        return False
    # Must have between 4 and 25 words
    words = s.split()
    if len(words) < 5 or len(words) > 25:
        return False
    return True


def sample_c4_clean_sentences():
    val_path = Path("data/processed/val.csv")
    if not val_path.exists():
        print(f"Error: {val_path} not found.")
        return

    clean_sentences = []
    seen = set()

    with open(val_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row.get("output", "").strip()
            if is_clean_sentence(target) and target not in seen:
                seen.add(target)
                clean_sentences.append(target)
                if len(clean_sentences) >= 500:
                    break

    print(f"Sampled {len(clean_sentences)} clean sentences.")

    out_path = Path("data/processed/c4_correct_500.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(clean_sentences, f, indent=2)

    print(f"Saved 500 clean sentences to {out_path}.")


if __name__ == "__main__":
    sample_c4_clean_sentences()
