"""
Phase M5: Production-Grade Hybrid Grammar Autocorrector Engine.

Features:
- Paragraph and sentence splitting (NLTK punkt)
- Protected spans shielding (URLs, emails, numbers, quoted text, code, @mentions)
- NLP analysis pipeline (preprocess -> tokenize -> POS tag)
- Multi-dictionary SymSpell spelling pass (frequency dictionary + bigram dictionary)
  skipping proper nouns, abbreviations, and protected spans
- CPU-friendly Seq2Seq model correction with beam search (num_beams=5), batched,
  torch.no_grad(), and thread count optimization
- Rule layer safety net from Phase 2 (subject-verb agreement, is/are, has/have, a/an)
- Two-pass iterative refinement (runs steps 3-4 second time only if changed)
- Strict safety gates (rejects if edit ratio > 0.5, length change > 40%, numbers,
  names, quotes, or valid nouns dropped) falling back to rule/spelling result
- Word-level edits via difflib: [{start, end, old, new, type, rule}] where type is
  spelling, grammar, or rule
- Thread-safe in-memory LRU cache
- 15-second request timeout with graceful fallback
- "Rules only" fallback if model weights are missing
- Startup log line stating active engine
"""

import os
import re
import html
import time
import difflib
import threading
import collections
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set
import concurrent.futures

import torch
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import importlib.resources
import symspellpy
from symspellpy import SymSpell, Verbosity

from app.nlp import analyze, preprocess, tokenize, pos_tag
from app.rules import apply_rules, Edit
from app.spelling import SPELLING_DICTIONARY, CONTEXTUAL_CONFUSABLES


# ---------------------------------------------------------------------------
# 1. Protected Spans Detection
# ---------------------------------------------------------------------------
PROTECTED_PATTERNS = [
    # URLs (http, https, www)
    re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE),
    # Emails
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # @mentions
    re.compile(r"@[A-Za-z0-9_]+"),
    # Inline code or code blocks
    re.compile(r"`[^`]+`"),
    # Quoted text (double, single, smart quotes)
    re.compile(r'"[^"]*"|\'[^\']*\'|“[^”]*”|‘[^’]*’'),
    # Numbers (integers, floats, comma-separated thousands)
    re.compile(r"\b\d+(?:[\.,]\d+)*\b"),
]


def find_protected_spans(text: str) -> List[Tuple[int, int, str]]:
    """
    Find all character ranges in text that must be protected from modification:
    URLs, emails, @mentions, code, quoted text, and numbers.
    """
    spans = []
    for pat in PROTECTED_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), m.group(0)))

    if not spans:
        return []

    # Sort and merge overlapping intervals
    spans.sort(key=lambda s: s[0])
    merged = [spans[0]]
    for cur in spans[1:]:
        p_start, p_end, p_text = merged[-1]
        if cur[0] < p_end:
            new_end = max(p_end, cur[1])
            merged[-1] = (p_start, new_end, text[p_start:new_end])
        else:
            merged.append(cur)
    return merged


def is_in_spans(start: int, end: int, spans: List[Tuple[int, int, str]]) -> bool:
    """Check if token interval [start, end] overlaps with any protected span."""
    for s_start, s_end, _ in spans:
        if max(start, s_start) < min(end, s_end):
            return True
    return False


# ---------------------------------------------------------------------------
# 2. In-Memory LRU Cache
# ---------------------------------------------------------------------------
class LRUCache:
    """Thread-safe in-memory LRU cache for grammar corrections."""

    def __init__(self, capacity: int = 1024):
        self.capacity = capacity
        self.cache: Dict[str, Dict[str, Any]] = collections.OrderedDict()
        self.lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                val = self.cache[key]
                return {
                    "original": val["original"],
                    "corrected": val["corrected"],
                    "tokens": list(val["tokens"]),
                    "pos_tags": list(val["pos_tags"]),
                    "edits": [dict(e) for e in val["edits"]],
                    "error_count": val["error_count"],
                    "engine": val["engine"],
                }
            return None

    def put(self, key: str, value: Dict[str, Any]):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = {
                "original": value["original"],
                "corrected": value["corrected"],
                "tokens": list(value["tokens"]),
                "pos_tags": list(value["pos_tags"]),
                "edits": [dict(e) for e in value["edits"]],
                "error_count": value["error_count"],
                "engine": value["engine"],
            }
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)


# ---------------------------------------------------------------------------
# 3. SymSpell Spelling Pass Manager
# ---------------------------------------------------------------------------
class SymSpellManager:
    """
    Manages frequency and bigram dictionaries for SymSpell spelling correction.
    Skips proper nouns, abbreviations, and protected spans.
    """

    def __init__(self):
        self.sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self._load_dictionaries()

    def _load_dictionaries(self):
        try:
            pkg_files = importlib.resources.files(symspellpy)
            dict_path = str(pkg_files / "frequency_dictionary_en_82_765.txt")
            bigram_path = str(pkg_files / "frequency_bigramdictionary_en_243_342.txt")
            self.sym.load_dictionary(dict_path, term_index=0, count_index=1)
            self.sym.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
            print(f"Loaded SymSpell dictionaries ({self.sym.word_count:,} words, {len(self.sym.bigrams):,} bigrams).", flush=True)
        except Exception as e:
            print(f"Warning: Failed to load SymSpell dictionaries: {e}", flush=True)

    def is_abbreviation(self, word: str) -> bool:
        """Check if word is an abbreviation (all caps len >= 2 or dotted)."""
        clean = re.sub(r"[^\w.]", "", word)
        if "." in clean and len(clean) >= 2:
            return True
        if clean.isupper() and len(clean) >= 2:
            return True
        return False

    def is_proper_noun(self, word: str, tag: str, is_first_word: bool) -> bool:
        """Check if token is a proper noun."""
        if tag in {"NNP", "NNPS"}:
            return True
        if not is_first_word and word and word[0].isupper():
            return True
        return False

    def correct_spelling(
        self,
        text: str,
        tokens: List[str],
        tags: List[str],
        spans: List[Tuple[int, int]],
        protected_spans: List[Tuple[int, int, str]],
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Run spelling pass with SymSpell + verified student misspellings dictionary.
        Returns corrected text and list of spelling edits.
        """
        if not text:
            return text, []

        # 1. Apply high-precision contextual confusable regexes
        intermediate = text
        for pat, repl in CONTEXTUAL_CONFUSABLES:
            intermediate = re.sub(pat, repl, intermediate)

        # 2. Word-level spelling correction
        edits = []
        word_iter = list(re.finditer(r"\b[A-Za-z']+\b", intermediate))
        for match in word_iter:
            w_start, w_end = match.start(), match.end()
            word = match.group(0)
            lw = word.lower()

            # Skip if inside protected span
            if is_in_spans(w_start, w_end, protected_spans):
                continue

            # Skip if abbreviation
            if self.is_abbreviation(word):
                continue

            # Check if proper noun
            is_first = (w_start == 0)
            is_prop = False
            for tok, tag, span in zip(tokens, tags, spans):
                if span[0] <= w_start and w_end <= span[1]:
                    if self.is_proper_noun(tok, tag, is_first):
                        is_prop = True
                    break
            if is_prop:
                continue

            # Check custom student misspellings dictionary
            replacement = None
            if lw in SPELLING_DICTIONARY:
                target = SPELLING_DICTIONARY[lw]
                if word.isupper() and len(word) > 1:
                    replacement = target.upper()
                elif word[0].isupper():
                    replacement = target[0].upper() + target[1:]
                else:
                    replacement = target
            elif lw not in self.sym.words:
                # Check SymSpell lookup
                suggestions = self.sym.lookup(lw, Verbosity.TOP, max_edit_distance=2)
                if suggestions:
                    sug_term = suggestions[0].term
                    if sug_term.lower() != lw:
                        if word.isupper() and len(word) > 1:
                            replacement = sug_term.upper()
                        elif word[0].isupper():
                            replacement = sug_term[0].upper() + sug_term[1:]
                        else:
                            replacement = sug_term

            if replacement and replacement != word:
                edits.append({
                    "start": w_start,
                    "end": w_end,
                    "old": word,
                    "new": replacement,
                    "type": "spelling",
                    "rule": "spelling_pass",
                })

        # Apply edits from right to left
        result_text = intermediate
        for e in sorted(edits, key=lambda x: x["start"], reverse=True):
            result_text = result_text[: e["start"]] + e["new"] + result_text[e["end"] :]

        return result_text, edits


# ---------------------------------------------------------------------------
# 4. Main Autocorrector Engine Class
# ---------------------------------------------------------------------------
class AutocorrectorEngine:
    """
    Production-Grade Hybrid Grammar Autocorrector Engine.
    Combines SymSpell spelling correction, deterministic linguistic rules,
    and a fine-tuned T5 Seq2Seq model with beam search and multi-pass refinement.
    """

    def __init__(self, model_dir: str = "model/gec-model"):
        self.model_dir = model_dir
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.engine_name = "rules only"
        self.cache = LRUCache(capacity=1024)
        self.symspell_manager = SymSpellManager()

        # Configure CPU thread count for friendly CPU inference
        cpu_threads = min(4, os.cpu_count() or 1)
        torch.set_num_threads(cpu_threads)

        self._load_model()
        print(f"Grammar Autocorrector active engine: {self.engine_name}", flush=True)

    def _load_model(self):
        """Load fine-tuned model and tokenizer once at startup if available."""
        path = Path(self.model_dir)
        has_adapter = (path / "adapter_config.json").exists()
        has_weights = (
            (path / "model.safetensors").exists()
            or (path / "pytorch_model.bin").exists()
        )
        has_config = (path / "config.json").exists()

        if has_adapter:
            try:
                from peft import PeftModel
                print(f"Loading base model 'grammarly/coedit-large' and LoRA adapter from {self.model_dir} on {self.device}...", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(str(path))
                base = AutoModelForSeq2SeqLM.from_pretrained(
                    "grammarly/coedit-large",
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                )
                self.model = PeftModel.from_pretrained(base, str(path))
                self.model.to(self.device)
                self.model.eval()
                self.engine_name = "rules+model"
                print("CoEdIT-Large LoRA model loaded successfully.", flush=True)
            except Exception as e:
                print(f"Warning: Failed to load LoRA adapter from {self.model_dir}: {e}. Falling back to rules only.", flush=True)
                self.model = None
                self.tokenizer = None
                self.engine_name = "rules only"
        elif has_weights and has_config:
            try:
                print(f"Loading GEC model from {self.model_dir} on {self.device}...", flush=True)
                self.tokenizer = AutoTokenizer.from_pretrained(str(path))
                self.model = AutoModelForSeq2SeqLM.from_pretrained(str(path))
                self.model.to(self.device)
                self.model.eval()
                self.engine_name = "rules+model"
                print("GEC model loaded successfully.", flush=True)
            except Exception as e:
                print(f"Warning: Failed to load model from {self.model_dir}: {e}. Falling back to rules only.", flush=True)
                self.model = None
                self.tokenizer = None
                self.engine_name = "rules only"
        else:
            print(f"No trained model found at {self.model_dir}. Running in rules-only mode.", flush=True)
            self.model = None
            self.tokenizer = None
            self.engine_name = "rules only"

    def _generate_model_batch(self, texts: List[str], num_beams: int = 5) -> List[str]:
        """Generate batched seq2seq corrections using beam search."""
        if self.model is None or self.tokenizer is None or not texts:
            return list(texts)

        prompt_prefix = "Fix the grammar: "
        formatted = [f"{prompt_prefix}{t}" for t in texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                formatted,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=128,
                num_beams=num_beams,
                early_stopping=True,
            )
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            del inputs, outputs
            return [d.strip() for d in decoded]

    def _should_reject_model(
        self,
        base_text: str,
        candidate_text: str,
        protected_spans: List[Tuple[int, int, str]],
    ) -> Tuple[bool, str]:
        """
        Safety Gates: Reject model candidate and fall back to rule/spelling output if:
        1. Length changes by more than 40%
        2. Edit ratio > 0.5 (difflib SequenceMatcher similarity < 0.5)
        3. Alters, drops, or invents numbers
        4. Alters, drops, or invents quotes or protected spans
        5. Alters, drops, or invents names / proper nouns
        6. Drops valid dictionary nouns (unwanted lexical substitution / hallucination)
        """
        # 1. Length check: > 40% change
        len_diff = abs(len(candidate_text) - len(base_text)) / max(1, len(base_text))
        if len_diff > 0.40:
            return True, f"length_change_exceeded ({len_diff:.1%})"

        # 2. Edit ratio check: > 50% rewritten
        similarity = difflib.SequenceMatcher(None, base_text, candidate_text).ratio()
        edit_ratio = 1.0 - similarity
        if edit_ratio > 0.50:
            return True, f"edit_ratio_exceeded ({edit_ratio:.1%})"

        # 3. Numbers check: must not drop, invent or change numbers
        nums_base = re.findall(r"\b\d+(?:[\.,]\d+)*\b", base_text)
        nums_cand = re.findall(r"\b\d+(?:[\.,]\d+)*\b", candidate_text)
        if nums_base != nums_cand:
            return True, f"numbers_altered ({nums_base} vs {nums_cand})"

        # 4. Quotes check: must not drop or invent quotes
        quotes_base = re.findall(r'"[^"]*"|\'[^\']*\'|“[^”]*”|‘[^’]*’', base_text)
        quotes_cand = re.findall(r'"[^"]*"|\'[^\']*\'|“[^”]*”|‘[^’]*’', candidate_text)
        if quotes_base != quotes_cand:
            return True, "quotes_altered"

        # 5. Names / Proper nouns check
        _, tokens, tags, _ = analyze(base_text)
        proper_nouns = [
            tokens[i].lower()
            for i, tag in enumerate(tags)
            if tag.startswith("NNP")
        ]
        cand_lower = candidate_text.lower()
        for name in proper_nouns:
            if name not in cand_lower:
                return True, f"name_altered ({name})"

        # 6. Protected spans check (URLs, emails, @mentions, code)
        for _, _, p_str in protected_spans:
            if p_str not in candidate_text:
                return True, f"protected_span_altered ({p_str})"

        # 7. Check dropped valid content nouns (e.g. university -> institution)
        for tok, tag in zip(tokens, tags):
            if tag.startswith("NN") and not tag.startswith("NNP"):
                lw = tok.lower()
                # Known dictionary word in input must not disappear completely
                if lw in self.symspell_manager.sym.words and lw not in cand_lower:
                    return True, f"noun_dropped ({lw})"

        return False, "accepted"

    def _build_word_level_edits(
        self,
        original_text: str,
        final_text: str,
        spelling_edits: List[Dict[str, Any]],
        rule_edits: List[Edit],
        global_offset: int,
    ) -> List[Dict[str, Any]]:
        """
        Build word-level edits using difflib between original and corrected text,
        classifying each edit type as: 'spelling', 'rule', or 'grammar'.
        """
        orig_tokens = list(re.finditer(r"\w+|[^\w\s]", original_text))
        final_tokens = list(re.finditer(r"\w+|[^\w\s]", final_text))

        orig_words = [m.group(0) for m in orig_tokens]
        final_words = [m.group(0) for m in final_tokens]

        matcher = difflib.SequenceMatcher(None, orig_words, final_words)
        edits = []

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag != "equal":
                if i1 < i2:
                    start = orig_tokens[i1].start()
                    end = orig_tokens[i2 - 1].end()
                    old_str = original_text[start:end]
                else:
                    start = orig_tokens[i1].start() if i1 < len(orig_tokens) else len(original_text)
                    end = start
                    old_str = ""

                if j1 < j2:
                    n_start = final_tokens[j1].start()
                    n_end = final_tokens[j2 - 1].end()
                    new_str = final_text[n_start:n_end]
                else:
                    new_str = ""

                # Determine edit type
                edit_type = "grammar"
                rule_name = "seq2seq_model"

                # Check if matches rule edit from Phase 2
                for r in rule_edits:
                    if old_str.strip().lower() == r.old.strip().lower():
                        edit_type = "rule"
                        rule_name = r.rule
                        break

                # Check if matches spelling edit
                for s in spelling_edits:
                    if old_str.strip().lower() == s["old"].strip().lower():
                        edit_type = "spelling"
                        rule_name = s.get("rule", "spelling_pass")
                        break

                edits.append({
                    "start": global_offset + start,
                    "end": global_offset + end,
                    "old": old_str,
                    "new": new_str,
                    "type": edit_type,
                    "rule": rule_name,
                })

        return edits

    def _correct_sentences_batched(
        self,
        sentences: List[str],
        sentence_offsets: List[int],
    ) -> Tuple[List[str], List[str], List[Tuple[str, str]], List[Dict[str, Any]]]:
        """
        Process a list of sentences belonging to a paragraph through the 7-step pipeline
        with batched model generation and safety gates.
        """
        all_tokens: List[str] = []
        all_tags: List[Tuple[str, str]] = []
        all_edits: List[Dict[str, Any]] = []
        corrected_sentences: List[str] = []

        if not sentences:
            return [], [], [], []

        # -------------------------------------------------------------------
        # Step 1 & 2: Preprocess, POS tag, Protected Spans, and Spelling Pass
        # -------------------------------------------------------------------
        sent_records = []
        for sent, offset in zip(sentences, sentence_offsets):
            clean_sent, tokens, tags, spans = analyze(sent)
            if not clean_sent:
                sent_records.append({
                    "original": sent,
                    "clean_sent": sent,
                    "tokens": [],
                    "tags": [],
                    "pos_pairs": [],
                    "spans": [],
                    "protected_spans": [],
                    "spelling_out": sent,
                    "spelling_edits": [],
                    "base_rule_edits": [],
                    "rule_spell_out": sent,
                    "current_text": sent,
                    "offset": offset,
                    "all_rule_edits": [],
                })
                continue

            pos_pairs = list(zip(tokens, tags))
            protected_spans = find_protected_spans(clean_sent)

            # Pre-calculate baseline Phase 2 rules
            base_rule_out, base_rule_edits = apply_rules(clean_sent)

            # Step 2: SymSpell spelling pass
            spelling_out, spelling_edits = self.symspell_manager.correct_spelling(
                clean_sent, tokens, tags, spans, protected_spans
            )

            # Pre-calculate rule layer on spelling output for fallback
            rule_spell_out, rule_spell_edits = apply_rules(spelling_out)

            sent_records.append({
                "original": sent,
                "clean_sent": clean_sent,
                "tokens": tokens,
                "tags": tags,
                "pos_pairs": pos_pairs,
                "spans": spans,
                "protected_spans": protected_spans,
                "spelling_out": spelling_out,
                "spelling_edits": spelling_edits,
                "base_rule_edits": base_rule_edits,
                "rule_spell_out": rule_spell_out,
                "current_text": spelling_out,
                "offset": offset,
                "all_rule_edits": list(base_rule_edits) + list(rule_spell_edits),
            })

        # -------------------------------------------------------------------
        # Step 3 & 4: Model Correction (Pass 1) + Rule Layer Safety Net
        # -------------------------------------------------------------------
        pass1_inputs = [r["current_text"] for r in sent_records]
        if self.model is not None and pass1_inputs:
            pass1_outputs = self._generate_model_batch(pass1_inputs, num_beams=5)
            for r, cand in zip(sent_records, pass1_outputs):
                if not r["clean_sent"]:
                    continue
                rejected, reason = self._should_reject_model(r["current_text"], cand, r["protected_spans"])
                if not rejected:
                    r["current_text"] = cand

        # Step 4: Rule layer from Phase 2 as safety net
        for r in sent_records:
            if not r["clean_sent"]:
                continue
            rule_out, r_edits = apply_rules(r["current_text"])
            r["all_rule_edits"].extend(r_edits)
            r["current_text"] = rule_out

        # -------------------------------------------------------------------
        # Step 5: Second Pass ONLY IF sentence still changed (max 2 passes)
        # -------------------------------------------------------------------
        changed_indices = [
            i for i, r in enumerate(sent_records)
            if r["clean_sent"] and r["current_text"] != r["spelling_out"]
        ]

        if self.model is not None and changed_indices:
            pass2_inputs = [sent_records[i]["current_text"] for i in changed_indices]
            pass2_outputs = self._generate_model_batch(pass2_inputs, num_beams=5)
            for idx, cand in zip(changed_indices, pass2_outputs):
                r = sent_records[idx]
                rejected, _ = self._should_reject_model(r["current_text"], cand, r["protected_spans"])
                if not rejected:
                    r["current_text"] = cand

            # Rule layer again for changed sentences
            for idx in changed_indices:
                r = sent_records[idx]
                rule_out, r_edits = apply_rules(r["current_text"])
                r["all_rule_edits"].extend(r_edits)
                r["current_text"] = rule_out

        # -------------------------------------------------------------------
        # Step 6 & 7: Safety Gates Verification & Word-Level Diffs
        # -------------------------------------------------------------------
        for r in sent_records:
            if not r["clean_sent"]:
                corrected_sentences.append(r["original"])
                continue

            # Step 6: Final safety gate check against original input
            rejected_final, _ = self._should_reject_model(
                r["clean_sent"], r["current_text"], r["protected_spans"]
            )
            if rejected_final:
                # Fall back to rule/spelling result
                r["current_text"] = r["rule_spell_out"]

            # Step 7: Build word-level edits with difflib
            edits = self._build_word_level_edits(
                r["clean_sent"],
                r["current_text"],
                r["spelling_edits"],
                r["all_rule_edits"],
                r["offset"],
            )

            corrected_sentences.append(r["current_text"])
            all_tokens.extend(r["tokens"])
            all_tags.extend(r["pos_pairs"])
            all_edits.extend(edits)

        return corrected_sentences, all_tokens, all_tags, all_edits

    def _correct_text_internal(self, text: str) -> Dict[str, Any]:
        """Internal text correction implementation across paragraphs and sentences."""
        cleaned_input = preprocess(text)
        if not cleaned_input:
            return {
                "original": text,
                "corrected": "",
                "tokens": [],
                "pos_tags": [],
                "edits": [],
                "error_count": 0,
                "engine": self.engine_name,
            }

        # Check in-memory cache
        cached = self.cache.get(cleaned_input)
        if cached is not None:
            return cached

        # Split into paragraphs to preserve layout and newlines
        paragraphs = text.split("\n")
        corrected_paragraphs = []
        all_tokens: List[str] = []
        all_tags: List[Tuple[str, str]] = []
        all_edits: List[Dict[str, Any]] = []

        global_offset = 0

        for para in paragraphs:
            para_clean = para.strip()
            if not para_clean:
                corrected_paragraphs.append("")
                continue

            # Split into sentences using NLTK punkt
            sentences = sent_tokenize(para_clean)
            if not sentences:
                sentences = [para_clean]

            sentence_offsets = []
            for sent in sentences:
                sent_offset = text.find(sent, global_offset)
                if sent_offset == -1:
                    sent_offset = global_offset
                sentence_offsets.append(sent_offset)
                global_offset = sent_offset + len(sent)

            # Run batched sentence pipeline
            cor_sents, p_toks, p_tags, p_edits = self._correct_sentences_batched(
                sentences, sentence_offsets
            )

            corrected_paragraphs.append(" ".join(cor_sents))
            all_tokens.extend(p_toks)
            all_tags.extend(p_tags)
            all_edits.extend(p_edits)

        final_corrected_text = "\n".join(corrected_paragraphs)

        result = {
            "original": cleaned_input,
            "corrected": final_corrected_text,
            "tokens": all_tokens,
            "pos_tags": all_tags,
            "edits": all_edits,
            "error_count": len(all_edits),
            "engine": self.engine_name,
        }

        # Store in LRU cache
        self.cache.put(cleaned_input, result)
        return result

    def correct_text(self, text: str, timeout: float = 15.0) -> Dict[str, Any]:
        """
        Public entry point with 15-second request timeout enforcement.
        Falls back gracefully to rules+spelling if timeout is reached.
        """
        # Short-circuit empty text
        if not text or not text.strip():
            return {
                "original": text,
                "corrected": "",
                "tokens": [],
                "pos_tags": [],
                "edits": [],
                "error_count": 0,
                "engine": self.engine_name,
            }

        # Check cache first (instant)
        cached = self.cache.get(preprocess(text))
        if cached is not None:
            return cached

        # Execute with thread pool executor for timeout enforcement
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._correct_text_internal, text)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print(f"Warning: Request exceeded {timeout}s timeout. Falling back to rule-only.", flush=True)
                clean_in = preprocess(text)
                rule_out, rule_edits = apply_rules(clean_in)
                edits = [
                    {
                        "start": 0,
                        "end": len(clean_in),
                        "old": e.old,
                        "new": e.new,
                        "type": "rule",
                        "rule": e.rule,
                    }
                    for e in rule_edits
                ]
                return {
                    "original": clean_in,
                    "corrected": rule_out,
                    "tokens": tokenize(clean_in),
                    "pos_tags": pos_tag(tokenize(clean_in)),
                    "edits": edits,
                    "error_count": len(edits),
                    "engine": "rules only (timeout fallback)",
                }


# Singleton engine instance loaded once at startup
_ENGINE = None


def get_engine() -> AutocorrectorEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = AutocorrectorEngine()
    return _ENGINE


def correct_text(text: str) -> Dict[str, Any]:
    """Top-level function for correcting text."""
    engine = get_engine()
    return engine.correct_text(text)
