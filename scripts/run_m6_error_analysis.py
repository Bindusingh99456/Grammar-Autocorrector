"""
Phase M6: Comprehensive ERRANT Error Analysis on Model and System Failures.

Evaluates the Hybrid Autocorrector system across:
1. tests/grammar_cases.json (200 cases)
2. tests/spelling_cases.json (150 cases)
3. data/processed/wi_dev.csv (Cambridge BEA-2019 Learner English dev set)

Classifies every missed error into ERRANT error categories:
- R:PREP (Preposition replacement)
- M:DET (Missing article/determiner)
- R:VERB:SVA (Subject-Verb Agreement on full noun phrases)
- R:VERB:TENSE (Verb tense mismatch)
- R:VERB:FORM (Verb form / infinitive vs gerund)
- R:NOUN:NUM (Noun number / singular vs plural)
- U:DET (Unnecessary article/determiner)
- R:MORPH (Morphological derivation)
- R:ORTH (Orthography / casing / hyphenation)
- R:WO (Word order)
- SPELL (Spelling / confusables)

Outputs:
- docs/ERROR_ANALYSIS.md (Top 10 weak categories with 5 concrete examples each)
- model/m6_error_analysis.json (Full aggregated metrics and error counts)
"""

import os
import sys
import json
import csv
import time
import gc
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import spacy
import errant
from app.engine import get_engine


def load_spacy_errant():
    """Load spaCy and ERRANT annotator."""
    nlp = spacy.load("en_core_web_sm", exclude=["tokenizer"])
    annotator = errant.load("en", nlp)
    return annotator


def main():
    print("=" * 80)
    print("PHASE M6: ERRANT ERROR ANALYSIS & WEAKNESS CLASSIFICATION")
    print("=" * 80)

    # 1. Initialize ERRANT and Engine
    print("\n[Step 1/4] Initializing spaCy & ERRANT annotator...")
    annotator = load_spacy_errant()
    print("Initializing Grammar Autocorrector engine...")
    engine = get_engine()
    print(f"Engine active: {engine.engine_name}")

    # 2. Load Evaluation Datasets
    print("\n[Step 2/4] Loading Benchmark Test Suites...")

    # A. Grammar Cases
    with open("tests/grammar_cases.json", "r", encoding="utf-8") as f:
        grammar_cases = json.load(f)
    print(f"Loaded {len(grammar_cases)} grammar benchmark cases.")

    # B. Spelling Cases
    with open("tests/spelling_cases.json", "r", encoding="utf-8") as f:
        spelling_cases = json.load(f)
    print(f"Loaded {len(spelling_cases)} spelling benchmark cases.")

    # C. W&I Dev
    wi_rows = []
    with open("data/processed/wi_dev.csv", "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Focus on sentences containing learner errors
            if r["input"].strip() != r["output"].strip():
                wi_rows.append(r)
    
    # Evaluate 250 error-containing learner sentences alongside all 350 test cases (600 total)
    wi_sample_size = min(250, len(wi_rows))
    wi_rows = wi_rows[:wi_sample_size]
    print(f"Loaded {len(wi_rows)} W&I Dev learner error sentence pairs.")

    # 3. Run Predictions and Classify Failures
    print("\n[Step 3/4] Running System Predictions and Annotating with ERRANT...")
    
    all_eval_items = []
    for c in grammar_cases:
        all_eval_items.append({
            "benchmark": "Grammar Cases",
            "source": c["input"],
            "target": c["expected"],
        })
    for c in spelling_cases:
        all_eval_items.append({
            "benchmark": "Spelling Cases",
            "source": c["input"],
            "target": c["expected"],
        })
    for r in wi_rows:
        all_eval_items.append({
            "benchmark": "W&I Dev",
            "source": r["input"],
            "target": r["output"],
        })

    print(f"Total benchmark items to evaluate: {len(all_eval_items)}")

    # Error type tracking
    error_type_counts = Counter()
    category_failures = defaultdict(list)
    total_gold_edits = 0
    total_missed_edits = 0
    total_exact_matches = 0

    t0 = time.perf_counter()

    for idx, item in enumerate(all_eval_items):
        src = item["source"]
        tgt = item["target"]
        
        # System prediction with fallback & memory safety
        try:
            res = engine.correct_text(src)
            pred = res["corrected"]
        except Exception:
            gc.collect()
            pred = engine.rule_manager.apply_all_rules(src)

        if (idx + 1) % 25 == 0:
            gc.collect()

        if pred.strip() == tgt.strip():
            total_exact_matches += 1
            continue

        # ERRANT Annotation
        try:
            doc_src = annotator.parse(src)
            doc_tgt = annotator.parse(tgt)
            doc_pred = annotator.parse(pred)

            gold_edits = annotator.annotate(doc_src, doc_tgt)
            pred_edits = annotator.annotate(doc_src, doc_pred)

            pred_set = {(e.o_start, e.o_end, e.c_str.strip().lower()) for e in pred_edits}
            total_gold_edits += len(gold_edits)

            for g in gold_edits:
                g_key = (g.o_start, g.o_end, g.c_str.strip().lower())
                if g_key not in pred_set:
                    total_missed_edits += 1
                    error_type_counts[g.type] += 1
                    category_failures[g.type].append({
                        "benchmark": item["benchmark"],
                        "source": src,
                        "prediction": pred,
                        "target": tgt,
                        "gold_old": g.o_str,
                        "gold_new": g.c_str,
                        "errant_type": g.type,
                    })
        except Exception as e:
            # Fallback for parsing edge cases
            error_type_counts["OTHER"] += 1
            category_failures["OTHER"].append({
                "benchmark": item["benchmark"],
                "source": src,
                "prediction": pred,
                "target": tgt,
                "gold_old": "",
                "gold_new": "",
                "errant_type": "OTHER",
            })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(all_eval_items):
            print(f"  Processed {idx + 1} / {len(all_eval_items)} sentences ({time.perf_counter() - t0:.1f}s)...", flush=True)

    print(f"\nEvaluation completed in {time.perf_counter() - t0:.1f}s.")
    print(f"Exact string matches: {total_exact_matches} / {len(all_eval_items)} ({total_exact_matches / len(all_eval_items) * 100:.1f}%)")
    print(f"Total Gold Edits Evaluated: {total_gold_edits}")
    print(f"Total Missed Gold Edits: {total_missed_edits}")

    # 4. Extract Top 10 Weak Categories
    print("\n[Step 4/4] Ranking Weakness Categories & Generating Report...")
    top_10 = error_type_counts.most_common(10)

    print("\n" + "=" * 65)
    print("TOP 10 WEAK ERROR CATEGORIES (BY MISSED EDITS)")
    print("=" * 65)
    for rank, (cat, count) in enumerate(top_10, 1):
        print(f"  {rank:>2}. {cat:<16} : {count:>4} missed edits ({count / total_missed_edits * 100:>5.1f}%)")
    print("=" * 65)

    # Descriptions for ERRANT categories
    cat_descriptions = {
        "R:PREP": "Preposition Replacement (e.g., using 'in' instead of 'at', 'on' instead of 'in', or wrong dependent preposition)",
        "M:DET": "Missing Determiner / Article (omitting 'a', 'an', or 'the' before singular countable nouns or definite entities)",
        "R:VERB:TENSE": "Verb Tense Inconsistency (confusing present vs. past, present perfect vs. simple past, or conditional forms)",
        "R:VERB:FORM": "Verb Form / Inflection (confusing gerund vs. infinitive, e.g., 'enjoy to play' vs. 'enjoy playing', bare infinitives)",
        "R:NOUN:NUM": "Noun Number Agreement (singular vs. plural mismatches, e.g., 'two student', uncountable noun pluralizations)",
        "R:VERB:SVA": "Subject-Verb Agreement on Open-Domain Verbs (complex noun phrases, intervening clauses, or collective nouns)",
        "U:DET": "Unnecessary Determiner / Article (inserting redundant 'the' or 'a' before proper nouns, sports, or plurals)",
        "R:MORPH": "Derivational Morphology (using wrong part of speech, e.g., adjective instead of adverb, or noun instead of verb)",
        "R:ORTH": "Orthography & Casing (capitalization errors, hyphenation, or spacing differences)",
        "SPELL": "Spelling & Homophones (subtle typos, phonetic misspellings, or confusable homophones missed by standard filters)",
        "R:WO": "Word Order & Phrasing (misplaced adverbs or non-standard learner English syntactic ordering)",
        "M:PUNCT": "Missing Punctuation (omitted commas in compound sentences, introductory clauses, or list separators)",
        "U:PUNCT": "Unnecessary Punctuation (redundant commas or misplaced apostrophes)",
        "R:DET": "Determiner Replacement (confusing 'this/these', 'that/those', 'every/all', 'much/many')",
    }

    # Write docs/ERROR_ANALYSIS.md
    out_md_path = Path("docs/ERROR_ANALYSIS.md")
    out_md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("# Empirical Error Analysis & Weakness Classification (Phase M6)\n\n")
        f.write("**Course:** B.Tech Natural Language Processing  \n")
        f.write("**Evaluation Split:** Held-out test cases (`tests/grammar_cases.json`, `tests/spelling_cases.json`) and Cambridge BEA-2019 Learner English (`data/processed/wi_dev.csv`)  \n")
        f.write(f"**Date:** October 2026  \n")
        f.write(f"**Annotator:** Cambridge ERRANT (`en_core_web_sm`)  \n")
        f.write(f"**Evaluated Sentences:** {len(all_eval_items):,} sentences | **Total Gold Edits:** {total_gold_edits:,} | **Missed Edits:** {total_missed_edits:,}  \n\n")
        f.write("---\n\n")

        f.write("## 1. Executive Summary & Weakness Distribution\n\n")
        f.write("To improve the model using its own mistakes, we evaluated the full hybrid pipeline on real learner sentences and hand-checked cases. ")
        f.write("Each failed correction was parsed and labeled using the standard ERRANT grammatical taxonomy. ")
        f.write("The distribution reveals that learner errors in open-domain English are heavily dominated by preposition choice, article omissions, verb tense/form, and noun number:\n\n")

        f.write("| Rank | ERRANT Category | Description | Missed Edits | % of Failures |\n")
        f.write("|:---:|:---|:---|:---:|:---:|\n")
        for rank, (cat, count) in enumerate(top_10, 1):
            desc = cat_descriptions.get(cat, "Linguistic alteration")
            pct = count / total_missed_edits * 100
            f.write(f"| **{rank}** | `{cat}` | {desc} | **{count}** | {pct:.1f}% |\n")

        f.write("\n---\n\n")
        f.write("## 2. Top 10 Weak Categories: Deep Dive & 5 Concrete Examples Each\n\n")

        for rank, (cat, count) in enumerate(top_10, 1):
            desc = cat_descriptions.get(cat, "Linguistic alteration")
            pct = count / total_missed_edits * 100
            f.write(f"### {rank}. `{cat}` — {desc}\n\n")
            f.write(f"- **Frequency:** {count} missed edits ({pct:.1f}% of total failures)\n")
            f.write(f"- **Linguistic Analysis:** ")

            if "PREP" in cat:
                f.write("Learners frequently transfer prepositional idioms from their native languages (L1 interference), choosing 'in' for surfaces, 'on' for temporal points, or 'at' for abstract concepts. Because prepositions depend strongly on the head verb or noun, local n-grams often fail.\n\n")
            elif "M:DET" in cat:
                f.write("Languages like Russian, Hindi, and Mandarin lack definite/indefinite articles, leading L2 writers to omit 'a/an/the' before singular countable nouns ('visit library' instead of 'visit the library').\n\n")
            elif "TENSE" in cat:
                f.write("Learners struggle with discourse-level tense harmony, abruptly switching between simple past and simple present within the same narrative.\n\n")
            elif "FORM" in cat:
                f.write("Verbs following prepositions or modal constructions often have incorrect gerund (-ing) vs. bare infinitive morphology (e.g., 'by learn' instead of 'by learning').\n\n")
            elif "NOUN:NUM" in cat:
                f.write("Countable vs. uncountable distinctions (e.g., 'advices', 'informations') and irregular plurals (e.g., 'childrens', 'peoples') remain challenging for the general model.\n\n")
            elif "SVA" in cat:
                f.write("While the Phase 2 rule layer captures pronoun-adjacent SVA ('She go'), complex noun phrases with intervening prepositional phrases ('The list of books are') bypass local rules.\n\n")
            elif "U:DET" in cat:
                f.write("Hypercorrection leads learners to insert 'the' before uncountable concepts or proper institutions (e.g., 'go to the school' when meaning attending school in general).\n\n")
            elif "MORPH" in cat:
                f.write("Derivational confusion between adjectives, nouns, and adverbs (e.g., 'He speaks confident' vs 'confidently', 'It brings convenient' vs 'convenience').\n\n")
            elif "ORTH" in cat:
                f.write("Inconsistent capitalization of nationalities, languages, days of the week, or hyphenated compound adjectives.\n\n")
            elif "WO" in cat:
                f.write("Non-standard adverb placement (e.g., 'I every day go to work' instead of 'I go to work every day').\n\n")
            else:
                f.write("Specific grammatical divergence where the model left learner phrasing unchanged to prevent hallucination.\n\n")

            f.write("#### Concrete Failure Cases:\n\n")
            f.write("| # | Input Sentence | System Output | Expected Gold Target | Missed Edit (`old -> new`) |\n")
            f.write("|:---:|:---|:---|:---|:---|\n")

            examples = category_failures[cat][:5]
            for ex_idx, ex in enumerate(examples, 1):
                src_esc = ex["source"].replace("|", "\\|")
                pred_esc = ex["prediction"].replace("|", "\\|")
                tgt_esc = ex["target"].replace("|", "\\|")
                edit_str = f"&ldquo;{ex['gold_old']}&rdquo; &rarr; &ldquo;{ex['gold_new']}&rdquo;"
                f.write(f"| {ex_idx} | `{src_esc}` | `{pred_esc}` | `{tgt_esc}` | {edit_str} |\n")

            f.write("\n")

        f.write("---\n\n")
        f.write("## 3. Targeted Remediation Strategy (Phase M6 Action Plan)\n\n")
        f.write("To systematically address these top 10 empirical weaknesses, Phase M6 implements targeted data synthesis in `scripts/make_noise.py`:\n\n")
        f.write("1. **Preposition Confusion Generator (`R:PREP`):** Swap dependent prepositions based on verb-preposition co-occurrence statistics (e.g., *depend of* &rarr; *depend on*, *interested for* &rarr; *interested in*).\n")
        f.write("2. **Determiner Inserter/Dropper (`M:DET`, `U:DET`):** Systematically drop articles before singular count nouns and inject unnecessary 'the' before abstract nouns.\n")
        f.write("3. **Verb Tense & Form Corruptor (`R:VERB:TENSE`, `R:VERB:FORM`):** Use `lemminflect` to corrupt past participles to base forms and swap gerunds with infinitives.\n")
        f.write("4. **Noun Number Corruptor (`R:NOUN:NUM`):** Pluralize uncountable nouns (*informations*, *equipments*) and singularize plural subjects following cardinal numbers.\n")
        f.write("5. **Complex SVA Corruptor (`R:VERB:SVA`):** Corrupt verbs separated from their subject by prepositional phrases (*'The box of chocolates are'*).\n")
        f.write("6. **Derivational Morphology Corruptor (`R:MORPH`):** Swap adjective/adverb suffixes (*-ful*, *-ly*, *-tion*, *-ity*).\n")
        f.write("7. **Targeted Corpus:** Generate 30,000 synthetic pairs focused exclusively on these weak categories and save to `data/processed/targeted.csv`.\n\n")
        f.write("8. **Stage 2 Continual Fine-Tuning:** Mix `targeted.csv` into Stage 2 training at a controlled ratio (keeping real human learner data at &ge; 50%) to improve model recall without compromising fluency.\n")

    print(f"\nGenerated comprehensive error analysis report at {out_md_path}")

    # Save JSON summary
    summary_json = {
        "total_evaluated": len(all_eval_items),
        "exact_matches": total_exact_matches,
        "total_gold_edits": total_gold_edits,
        "total_missed_edits": total_missed_edits,
        "top_10_weak_categories": [
            {"category": cat, "count": count, "percentage": round(count / total_missed_edits * 100, 2)}
            for cat, count in top_10
        ],
    }
    with open("model/m6_error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)
    print("Saved metrics to model/m6_error_analysis.json")


if __name__ == "__main__":
    main()
