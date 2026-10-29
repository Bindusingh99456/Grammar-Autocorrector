# Verification & Evaluation Results

**Project:** Grammar Autocorrector  
**Course:** B.Tech Natural Language Processing  
**Student:** Bindu Singh M (SRN: `R24EJ024`)  
**Institution:** REVA University, Bengaluru  
**Date:** October 2026  

---

## 1. Executive Summary

This document presents the verified empirical metrics, test results, contamination audits, and live endpoint validation for the **Grammar Autocorrector** application.

- **Automated Test Suite (Pytest):** 143 / 143 tests passing across unit rules, API validation, web pages, and test set isolation.
- **Targeted Benchmark Suite (`tests/cases.json`):** 110 / 110 regression cases passing on targeted agreement patterns and invariant protections.
- **Data Contamination Audit (`scripts/check_overlap.py`):** Verified **0 overlapping sentences** between the training dataset (276,337 pairs) and both evaluation benchmarks (747 JFLEG test sentences + 4,187 W&I+LOCNESS dev sentences).
- **JFLEG Test Split Evaluation (`model/evaluate.py`):** Measured on all 747 test sentences against 4 human references each; achieved a Corpus GLEU score of **0.7755** (baseline: **0.7732**), modifying **51.94%** of sentences.
- **Live Endpoint Verification (`scripts/verify_live_server.py`):** 12 / 12 HTTP endpoints and interactive forms returned expected status codes and content.

---

## 2. Automated Test Suite (Pytest)

The automated test suite verifies preprocessing, linguistic morphology, API validation, Jinja2 template rendering, static asset constraints, and data isolation.

```
platform win32 -- Python 3.14.5, pytest-9.1.1
collected 143 items

tests/test_rules.py .................................................... [ 36%]
......................................................                   [ 74%]
tests/test_api.py .............................                          [ 94%]
tests/test_overlap.py .                                                  [ 95%]
tests/test_pages.py .......                                              [100%]

============================ 143 passed in 31.42s =============================
```

### Test Suite Breakdown

| Test Module | Tests | Status | Focus Areas |
|---|---|---|---|
| `tests/test_rules.py` | 106 | 106 Passed | 8 project report cases, 40+ inflection cases (-s, -es, -ies, irregulars), phonetic a/an (silent-h, yu-sound, wa-sound), modals, past-tense guards, quoted text, proper nouns. |
| `tests/test_api.py` | 29 | 29 Passed | Health check, GET `/`, POST `/correct`, empty input rejection, length limit (>2000 chars), 8 report cases via REST API, 15 regression cases. |
| `tests/test_overlap.py` | 1 | 1 Passed | Programmatic assertion confirming 0 training sentence overlap with JFLEG test and W&I+LOCNESS dev. |
| `tests/test_pages.py` | 7 | 7 Passed | Static assets (`style.css`, `app.js` < 40 lines), empty state rendering, inline `<del>` and `<ins>` diffs, 8-step SVG pipeline, test cases table with Run It buttons, About page credits, forbidden buzzword checks. |
| **Total** | **143** | **143 Passed** | **All Automated Tests Verified** |

---

## 3. Targeted Benchmark Suite (`tests/cases.json`)

To verify correctness across targeted grammatical patterns and ensure valid sentences remain unchanged, a benchmark suite of 110 sentences was evaluated via `TestClient`.

### Category Breakdown Table

| Category | Description | Total | Passed | Failed | Pass Rate |
|---|---|---|---|---|---|
| `subject_verb_agreement` | he/she/it + base verb &rarr; 3rd person; plural pronouns + 3rd person &rarr; base form | 28 | 28 | 0 | 100.0% |
| `is_are` | Auxiliary mismatch (singular with 'are' &rarr; 'is'; plural with 'is' &rarr; 'are') | 6 | 6 | 0 | 100.0% |
| `has_have` | Verb agreement (3rd person singular with 'have' &rarr; 'has'; others with 'has' &rarr; 'have') | 7 | 7 | 0 | 100.0% |
| `does_do` | Auxiliary agreement in negative/question forms ('he do' &rarr; 'he does', 'they does' &rarr; 'they do') | 7 | 7 | 0 | 100.0% |
| `a_an` | Phonetic article selection (silent-h 'an hour', 'yu' sound 'a university', 'wa' sound 'a one-time') | 16 | 16 | 0 | 100.0% |
| `correct_no_change` | Correct sentences verifying precision (no false positives) | 23 | 23 | 0 | 100.0% |
| `modal_verbs` | Sentences with modals (can, will, should, must) ensuring base verbs are never inflected | 8 | 8 | 0 | 100.0% |
| `past_tense` | Past tense verbs (went, saw, played, wrote, ate) ensuring past forms are preserved | 7 | 7 | 0 | 100.0% |
| `quotes_protection` | Single, double, and smart quoted text preserved verbatim without edits | 4 | 4 | 0 | 100.0% |
| `proper_nouns` | Proper nouns (Microsoft, London, REVA University, Google) shielded from erroneous agreement | 4 | 4 | 0 | 100.0% |
| **TOTAL** | **All Categories Combined** | **110** | **110** | **0** | **100.0%** |

---

## 4. Dataset Overlap & Contamination Audit

An automated contamination check was executed via `scripts/check_overlap.py` to ensure that evaluation benchmarks are strictly held out from training:

| Benchmark Split | Sentences Audited | Overlapping Rows Detected | Contamination Status |
|---|---|---|---|
| **JFLEG Test Split** (`test.src` + 4 references) | 747 | 0 | **Clean (0.0% Leakage)** |
| **W&I+LOCNESS Dev Split** (Cambridge BEA-2019) | 4,187 | 0 | **Clean (0.0% Leakage)** |
| **Training Dataset** (`train.csv`) | 276,337 | 0 | **Fully Isolated** |

---

## 5. JFLEG Test Benchmark Evaluation

The fine-tuned sequence-to-sequence model (`model/gec-model`) was evaluated on the complete JFLEG test set using `model/evaluate.py`. Scores were computed using NLTK's `corpus_gleu` against all 4 human reference corrections per test sentence.

### Official Metrics (`model/evaluation_results.json`)

| Metric | Baseline (Uncorrected Input) | Fine-Tuned Model | Delta |
|---|---|---|---|
| **Corpus GLEU Score** | **0.7732** | **0.7755** | **+0.0023** |
| **Sentences Changed** | 0.00% (0 / 747) | 51.94% (388 / 747) | +51.94% |
| **Total Test Sentences** | 747 | 747 | &mdash; |
| **Human References / Sentence** | 4 | 4 | &mdash; |

### Sample Predictions from JFLEG Evaluation

1. **Source:** `New and new technology has been introduced to the society .`  
   **Prediction:** `New and new technology has been introduced to the society .`  
   **Ref 0:** `New technology has been introduced to society .`  

2. **Source:** `One possible outcome is that an environmentally-induced reduction in motorization levels in the richer countries will outweigh any rise in motorization levels in the poorer countries .`  
   **Prediction:** `One possible outcome is that an environmentally-induced reduction in motorization levels in the richer countries will outweigh any rise in motorization levels in the poorer countries .`  
   **Ref 0:** `One possible outcome is that an environmentally-induced reduction in motorization levels in richer countries will outweigh any rise in motorization levels in poorer countries .`  

3. **Source:** `Every person needs to know a bit about math , sciences , arts , literature and history in order to stand out in society .`  
   **Prediction:** `Every person needs to know a bit about math , sciences , arts , literature and history in order to stand out in society .`  
   **Ref 0:** `Every person needs to know a bit about math , science , art , literature and history in order to stand out in society .`  

---

## 6. Live Server Endpoint & Form Verification

All pages and interactive actions were tested over HTTP using `httpx` against the running Uvicorn server on `http://127.0.0.1:8000`:

| Endpoint / Interaction | Method | Expected Status | Actual Status | Verification Details | Result |
|---|---|---|---|---|---|
| Home Editor Page (`/`) | GET | 200 | 200 | Verified brand, sub-text, side-by-side grid, char counter | **PASS** |
| Editor Form Submission | POST | 200 | 200 | Submitted `"She go to college every day."`; verified `"She goes to college every day."`, inline `<del>`/`<ins>`, rule badge | **PASS** |
| Empty Form Submission | POST | 400 | 400 | Submitted `""`; verified `"Please enter some text to correct."` error alert | **PASS** |
| Excessive Input (>2000 chars) | POST | 400 | 400 | Submitted 2,500 characters; verified limit rejection message | **PASS** |
| Test Case 'Run it' Action | POST | 200 | 200 | Submitted `"He have a car."`; verified `"He has a car."` and diff rendering | **PASS** |
| How It Works (`/how-it-works`) | GET | 200 | 200 | Verified 8-step pipeline SVG flowchart and worked example trace | **PASS** |
| Test Cases (`/test-cases`) | GET | 200 | 200 | Verified 8 report cases table with individual "Run it" action buttons | **PASS** |
| About Page (`/about`) | GET | 200 | 200 | Verified student name (Bindu Singh M, R24EJ024), REVA University, tools, and credits | **PASS** |
| API Health (`/api/health`) | GET | 200 | 200 | Verified JSON response `{"status": "ok", "engine": "rules+model", "model_loaded": true}` | **PASS** |
| API Correct (`/api/correct`) | POST | 200 | 200 | Submitted `{"text": "They is playing football."}`; verified JSON corrected text and edit offsets | **PASS** |
| Static Stylesheet (`/static/style.css`) | GET | 200 | 200 | Verified CSS variables (`--navy-primary: #0e2240;`), font imports, and responsive media queries | **PASS** |
| Static Script (`/static/app.js`) | GET | 200 | 200 | Verified clipboard copy, Ctrl+Enter handler, and length (35 lines, strictly < 40) | **PASS** |

---

## 7. Known Failures & Linguistic Boundaries

In academic reporting, documenting genuine linguistic edge cases is essential. The following failure modes and limitations are documented:

1. **Long-Distance Relative Clauses:**
   - *Example:* `"The teacher who prepares five different lessons every morning walk to school."`
   - *Failure Mode:* The intervening clause separates the singular subject `"teacher"` from the verb `"walk"` by 8 tokens, bypassing adjacent token inspection windows.

2. **Noun-Verb POS Homonym Ambiguity:**
   - *Example:* `"Her daily records show remarkable progress."`
   - *Failure Mode:* Isolated perceptual taggers can misclassify `"records"` as `VBZ` (3rd-person singular verb) instead of `NNS` (plural noun), triggering an erroneous agreement flag unless shielded.

3. **Subjunctive Mood:**
   - *Example:* `"I demand that he go immediately."`
   - *Failure Mode:* Mandative subjunctive grammar requires a base-form verb with third-person subjects, which simple agreement rules would incorrectly inflect to `"goes"`.

4. **Contextual Homophone Substitutions:**
   - *Example:* `"Their going to the station soon."`
   - *Failure Mode:* Homophones such as *"their"* vs *"they're"* are valid words in English dictionaries. Detecting contextual misselection requires full semantic language modeling rather than morphological rule replacement.

5. **Collective Noun Dialectal Differences:**
   - *Example:* `"The committee are divided on the decision."`
   - *Failure Mode:* British English accepts plural verb agreement for collective nouns, whereas American English typically requires singular verbs.

6. **CPU Inference Latency:**
   - Deterministic linguistic rules execute in <1.5 ms per sentence, while neural seq2seq inference on a laptop CPU requires ~250–350 ms per sentence.
