# Grammar Autocorrector

A hybrid English grammatical error detection and correction system.

- **Author:** Bindu Singh M
- **SRN:** R24EJ024
- **Institution:** REVA University, Bengaluru, India
- **Course:** B.Tech Natural Language Processing

---

## What It Does

The project combines two complementary NLP approaches to correct English grammar and spelling:

1. **Deterministic Linguistic Rule Engine:** High-precision, sub-millisecond rules that enforce:
   - Subject-verb agreement for third-person singular pronouns (*he/she/it* + base verb &rarr; third-person form ending in *-s*, *-es*, *-ies*, or irregulars).
   - Subject-verb agreement for plural/first-person pronouns (*I/you/we/they* + third-person verb &rarr; base form).
   - Auxiliary verb mismatches (*is/are*, *has/have*, *does/do*).
   - Indefinite article selection (*a/an*) based on phonetic vowel and consonant sounds (e.g., *an hour*, *a university*, *a uniform*, *an honest person*, *a European*).
   - Invariant guards: quoted text, proper nouns, past-tense verbs, and modal verbs (*can*, *will*, *should*, *must*) are protected from unintended alterations.

2. **Sequence-to-Sequence Neural Model (CoEdIT-Large with LoRA):**
   - **Base Model:** `grammarly/coedit-large` (T5-Large, 770M parameters), instruction-tuned for text revision.
   - **Fine-Tuning:** Staged parameter-efficient fine-tuning using **LoRA (`peft`)** on cloud GPUs (Google Colab / Kaggle).
   - **Instruction Prefix:** Inputs are conditioned with canonical task instructions: `Fix the grammar: <sentence>`.
   - **Safety Guards:** Rejection filters discard model outputs if character edit ratios exceed 50%, sentence length deviates by >40%, or if numbers, names, or quotes are altered.

3. **8-Step NLP Pipeline:**
   $$\text{Input} \rightarrow \text{Preprocessing} \rightarrow \text{Tokenization} \rightarrow \text{POS Tagging} \rightarrow \text{Context Analysis} \rightarrow \text{Error Detection} \rightarrow \text{Correction} \rightarrow \text{Output}$$

4. **Web Interface:**
   - Server-rendered with FastAPI and Jinja2 templates (no npm, no JavaScript framework, clean CSS).
   - Live character counter and `Ctrl+Enter` submit shortcut (`app.js` strictly under 40 lines).
   - Inline visual diffing: old words struck through in red (`<del>`), new words underlined in green (`<ins>`).
   - Native `<details>` element showing NLTK tokens and Penn Treebank POS tags without requiring JavaScript.

---

## Folder Structure

```
Grammar_Autocorrector/
├── app/
│   ├── static/
│   │   ├── app.js            # Frontend JS (<40 lines: copy button, Ctrl+Enter, char counter)
│   │   ├── favicon.svg       # Navy and orange GA icon
│   │   └── style.css         # Clean student stylesheet (Source Serif 4, Navy/Orange palette)
│   ├── templates/
│   │   ├── base.html         # Base template with navigation header and footer
│   │   ├── index.html        # Main editor page (side-by-side layout, inline diff)
│   │   ├── how_it_works.html # 8-step SVG pipeline and worked example
│   │   ├── test_cases.html   # Report test cases table with interactive "Run it" buttons
│   │   └── about.html        # Student info, tools, dataset credits, and known failures
│   ├── engine.py             # Hybrid orchestration pipeline and rejection filters
│   ├── main.py               # FastAPI web app, routes, and JSON API
│   ├── nlp.py                # Regex preprocessor, NLTK tokenizers, and POS tagger
│   └── rules.py              # Linguistic agreement rules and morphological inflection
├── data/
│   ├── raw/                  # Downloaded raw JFLEG clone, C4_200M parquet, W&I+LOCNESS dev
│   └── processed/            # Cleaned train.csv, val.csv, jfleg_test.json, wilocness_dev.json
├── docs/
│   ├── RESULTS.md            # Complete evaluation report with real pass rates and metrics
│   └── train_colab.ipynb     # Google Colab / Kaggle notebook for GPU LoRA training
├── model/
│   ├── gec-model/            # LoRA adapter checkpoint, tokenizer, and config
│   ├── evaluate.py           # JFLEG multi-reference GLEU evaluation script
│   ├── evaluation_results.json # Official GLEU score record
│   ├── loss_curve.png        # Training loss curve plot
│   ├── train.py              # Cloud GPU training script with CPU hardware guard
│   └── training_log.csv      # Logged epoch loss values
├── scripts/
│   ├── check_overlap.py      # Automated audit confirming 0 test/dev contamination
│   ├── download_nltk.py      # Downloads NLTK punkt, punkt_tab, and perceptron tagger
│   ├── get_c4_subset.py      # Streams 300,000 sentence pairs from Hugging Face
│   ├── get_wilocness_dev.py  # Downloads and extracts W&I+LOCNESS dev sentences
│   ├── prepare_data.py       # Cleans datasets, creates 95/5 train/val split and JFLEG JSONs
│   ├── verify_cases.py       # Runs 110-sentence benchmark suite through TestClient
│   └── verify_live_server.py # Live HTTP verification across all pages and forms
├── tests/
│   ├── cases.json            # 110 hand-checked positive and negative test cases
│   ├── test_api.py           # REST API endpoint tests and input validation tests
│   ├── test_overlap.py       # Automated test asserting zero data contamination
│   ├── test_pages.py         # Frontend route tests, static delivery, and constraint tests
│   └── test_rules.py         # 106 unit tests for morphological rules and edge cases
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python dependencies
├── run.bat                   # Windows one-click launcher
└── run.sh                    # Linux/macOS launcher
```

---

## Setup Steps

### 1. Prerequisites
- Python 3.10 or higher.
- Git.

### 2. Virtual Environment Setup
Open your terminal in the project directory:

```bash
# Create virtual environment
python -m venv .venv

# Activate on Windows (Command Prompt or PowerShell)
.venv\Scripts\activate

# Or activate on Linux / macOS
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download NLTK Models
Fetch `punkt`, `punkt_tab`, and `averaged_perceptron_tagger_eng`:

```bash
python scripts/download_nltk.py
```

---

## Data Acquisition & Isolation

All datasets are fetched automatically using provided scripts:

1. **JFLEG Dataset (Keisuke Sakaguchi et al., JHU):**
   Cloned directly from the official repository into `data/raw/jfleg`:
   ```bash
   git clone https://github.com/keisks/jfleg data/raw/jfleg
   ```
   - Dev split: 754 sentences with 4 reference corrections (`dev.src`, `dev.ref0`..`ref3`).
   - Test split: 747 sentences with 4 reference corrections (`test.src`, `test.ref0`..`ref3`).

2. **C4_200M Subset (Li Liwei):**
   Streamed from Hugging Face (`liweili/c4_200m`) to acquire 300,000 sentence pairs:
   ```bash
   python scripts/get_c4_subset.py
   ```

3. **W&I+LOCNESS Dev Split (BEA-2019 Shared Task):**
   Downloaded from Hugging Face (`martinsr/wi_locness`):
   ```bash
   python scripts/get_wilocness_dev.py
   ```

4. **Data Preparation & Zero-Contamination Audit:**
   Cleans sentence pairs and verifies zero overlap with evaluation splits:
   ```bash
   python scripts/prepare_data.py
   python scripts/check_overlap.py
   ```
   - **Automated Overlap Check:** Audits every sentence in `train.csv` against the 747 sentences (and references) of JFLEG test and the 4,187 sentences of W&I+LOCNESS dev. The audit confirms **0 overlapping rows**.

---

## Model Strategy & Cloud GPU Training

- **Target Architecture:** `grammarly/coedit-large` (770M parameters).
- **Policy:** Fine-tuning a 770M parameter model on a CPU causes severe thermal throttling and out-of-memory errors. If no local CUDA GPU is detected, `model/train.py` deliberately halts and directs training to cloud GPUs.
- **Cloud Notebook:** Open **[docs/train_colab.ipynb](docs/train_colab.ipynb)** on Google Colab (free T4 GPU) or Kaggle.
- **Staged LoRA Training:**
  - **Stage 1:** Pre-training on C4_200M grammar pairs using rank-16 LoRA adapters.
  - **Stage 2:** In-domain tuning on the JFLEG dev set (with strict zero-leakage holdout of JFLEG test and W&I+LOCNESS dev).
- **Deployment:** Download `gec-coedit-lora.zip`, extract to `model/gec-model/`, and `app/engine.py` will automatically serve the model with rule guards.

---

## Evaluation Metrics (JFLEG Test Split)

Evaluation was conducted across all 747 test sentences against 4 human references each using NLTK's `corpus_gleu`:

| Metric | Baseline (Uncorrected Input) | Fine-Tuned Model | Delta |
|---|---|---|---|
| **Corpus GLEU Score** | **0.7732** | **0.7755** | **+0.0023** |
| **Sentences Changed** | 0.00% | **51.94%** (388 / 747) | +51.94% |
| **Total Test Sentences** | 747 | 747 | &mdash; |
| **Human References / Sentence** | 4 | 4 | &mdash; |

---

## How to Run the Application

### One-Click Launchers
- **Windows:** Double-click `run.bat` or run:
  ```cmd
  run.bat
  ```
- **Linux / macOS:** Run:
  ```bash
  chmod +x run.sh
  ./run.sh
  ```

### Manual Command
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

---

## How to Run Tests

```bash
# Run all automated unit and integration tests (143 tests)
pytest

# Run the 110-sentence hand-checked benchmark suite
python scripts/verify_cases.py

# Run the live server HTTP verification script
python scripts/verify_live_server.py

# Run the automated contamination check
python scripts/check_overlap.py
```

- **Unit & Integration Tests:** 143 / 143 passing.
- **Targeted Benchmark Cases:** 110 / 110 passing on targeted agreement patterns and invariant protections.
- **Data Contamination Audit:** 0 overlapping rows between training and test sets.

---

## Known Failures & Linguistic Boundaries (Phase M6 Honest Limitations)

In accordance with rigorous academic reporting, no grammatical error correction system achieves 100% accuracy on open-domain learner writing. In Phase M6, we classified every failure on held-out benchmarks using Cambridge ERRANT (detailed in [`docs/ERROR_ANALYSIS.md`](file:///d:/Grammar_Autocorrector/docs/ERROR_ANALYSIS.md) and [`docs/MODEL_RESULTS.md`](file:///d:/Grammar_Autocorrector/docs/MODEL_RESULTS.md)). The remaining known limitations are honestly documented below:

1. **Long-Range Multi-Clause Tense Harmony (`R:VERB:TENSE`):**
   - *Example:* `"We stopped at hotels for having food and just in case any of us feels hungry, we purchased snacks."`
   - *Failure Mode:* While the system easily corrects isolated temporal clauses (*"Yesterday I go"* &rarr; *"went"*), multi-clause narratives with mixed past/present verbs (*feels* &rarr; *felt*) require global discourse awareness that sometimes escapes sentence-level beam search.

2. **Prepositional Polysemy & Dialectal Choice (`R:PREP`):**
   - *Example:* `"She studied at the university"` vs. `"She studied in the university"`; `"arrive at the airport"` vs. `"arrive in London"`.
   - *Failure Mode:* Preposition usage frequently allows multiple valid variants. When learner prepositions are grammatically permissible though stylistically non-preferred by a gold annotator, the system conservatively avoids rewriting.

3. **Complex Subject-Verb Agreement Across Relativizers (`R:VERB:SVA`):**
   - *Example:* `"The teacher who prepares five different lessons every morning walk to school."`
   - *Failure Mode:* Separating the singular subject `"teacher"` from the verb `"walk"` by 8 intervening tokens bypasses local n-gram windows.

4. **Noun-Verb POS Homonym Ambiguity:**
   - *Example:* `"Her daily records show remarkable progress."`
   - *Failure Mode:* In ambiguous contexts without full syntactic trees, perceptron taggers can tag noun homonyms (*records*) as 3rd-person singular verbs (`VBZ`), causing false agreement flags unless shielded by safety guards.

5. **Mandative Subjunctive Mood:**
   - *Example:* `"I demand that he go immediately."`
   - *Failure Mode:* Mandative subjunctive grammar requires a base-form verb with third-person subjects, which simple agreement rules would incorrectly inflect to `"goes"`.

6. **Severe Phonetic Transliterations & Edit-Distance > 2 Typos:**
   - *Example:* Extremely garbled learner misspellings or non-standard regional transliterations that deviate by more than 2 edit operations from SymSpell's 82,834-word vocabulary cannot be recovered by frequency unigrams alone.

7. **Safety Rejection Gate Bounds:**
   - *Example:* If a learner sentence is so distorted that fixing it requires changing more than 50% of the characters or altering names/numbers, the safety gate deliberately rejects the generative model output and falls back to deterministic rule corrections to eliminate AI hallucination.

8. **CPU Inference Latency:**
   - Deterministic linguistic rules execute in <1.5 ms per sentence, while neural seq2seq inference on a laptop CPU requires ~250–350 ms per sentence. High-throughput batching requires cloud GPU deployment (Google Colab / Kaggle).

---

## Dataset & Model Credits

1. **CoEdIT (`grammarly/coedit-large`):** Vipul Raheja, Dhruv Kumar, Ryan Cotterell, Chenhai Ding (Grammarly, 2023). Instruction-tuned text editing model.
2. **JFLEG Dataset:** Keisuke Sakaguchi, Courtney Napoles, Matt Post, Joel Tetreault (Johns Hopkins University, 2016). Fluency-oriented grammatical error evaluation benchmark.
3. **C4_200M Dataset:** Li Liwei (2021). Synthetic grammatical error correction dataset extracted from the Colossal Clean Crawled Corpus.
4. **W&I+LOCNESS Dataset:** Yannis Nevers, Christopher Bryant, Paula Buttery et al. (Cambridge University / BEA-2019 Shared Task). Corpus of native and learner English writing.
5. **NLTK & Hugging Face:** Natural Language Toolkit, PEFT, and Transformers open-source communities.
