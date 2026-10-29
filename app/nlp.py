import re
from typing import List, Tuple
import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.tag import pos_tag as nltk_pos_tag


_TOKENIZER = TreebankWordTokenizer()


def preprocess(text: str) -> str:
    """
    Clean and normalize input text by collapsing whitespace
    and stripping leading/trailing spaces.
    """
    if not text:
        return ""
    # Normalize unicode spaces to standard space
    cleaned = re.sub(r"[\r\t\f\v ]+", " ", text)
    # Collapse multiple newlines or spaces
    cleaned = re.sub(r"\n\s+", "\n", cleaned)
    return cleaned.strip()


def tokenize(text: str) -> List[str]:
    """
    Tokenize text using NLTK's Treebank tokenizer.
    """
    clean_text = preprocess(text)
    if not clean_text:
        return []
    return _TOKENIZER.tokenize(clean_text)


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize text and return each token alongside its start and end
    character offsets in the preprocessed text.
    """
    clean_text = preprocess(text)
    if not clean_text:
        return []

    spans = list(_TOKENIZER.span_tokenize(clean_text))
    tokens_with_spans = []
    for start, end in spans:
        tokens_with_spans.append((clean_text[start:end], start, end))
    return tokens_with_spans


def pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Assign Penn Treebank part-of-speech tags to tokens using NLTK.
    """
    if not tokens:
        return []
    return nltk_pos_tag(tokens)


def analyze(text: str) -> Tuple[str, List[str], List[str], List[Tuple[int, int]]]:
    """
    Run preprocessing, tokenization, POS tagging, and span extraction.
    Returns:
        clean_text: normalized text
        tokens: list of token strings
        tags: list of POS tag strings
        spans: list of (start_char, end_char) offsets in clean_text
    """
    clean_text = preprocess(text)
    if not clean_text:
        return "", [], [], []

    spans = list(_TOKENIZER.span_tokenize(clean_text))
    tokens = [clean_text[start:end] for start, end in spans]
    tagged = pos_tag(tokens)
    tags = [tag for _, tag in tagged]
    return clean_text, tokens, tags, spans
