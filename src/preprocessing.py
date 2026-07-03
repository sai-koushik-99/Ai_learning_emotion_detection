"""
Text preprocessing for emotion detection.

Provides:
- validate_input()        : length / empty guard
- clean_text()            : lowercase, URL/mention/HTML strip
- tokenize_and_lemmatize(): NLTK tokenise → stopword filter → lemmatise
- preprocess_for_bilstm() : cleaned + lemmatised string for Keras tokeniser
- preprocess_for_bert()   : lightly cleaned string for DistilBERT tokeniser
- full_preprocess()       : convenience wrapper returning both variants
"""

import re
import string
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── NLTK data bootstrap ────────────────────────────────────────────────────
for _pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(
            f"tokenizers/{_pkg}" if "punkt" in _pkg else f"corpora/{_pkg}"
        )
    except LookupError:
        nltk.download(_pkg, quiet=True)

_LEMMATIZER = WordNetLemmatizer()
_STOPWORDS  = set(stopwords.words("english"))
# Keep negations — they carry strong sentiment signal
_KEEP_WORDS = {"no", "not", "never", "nor", "neither", "hardly", "barely"}
_STOPWORDS -= _KEEP_WORDS

MAX_INPUT_CHARS = 2000

# ── Emotion classes (single source of truth used across the whole project) ─
EMOTION_CLASSES: List[str] = [
    "Confused", "Frustrated", "Curious",
    "Confident", "Excited", "Anxious", "Neutral",
]
NUM_CLASSES = len(EMOTION_CLASSES)
LABEL2IDX   = {e: i for i, e in enumerate(EMOTION_CLASSES)}
IDX2LABEL   = {i: e for i, e in enumerate(EMOTION_CLASSES)}

# ── Mixed-emotion threshold ────────────────────────────────────────────────
MIXED_EMOTION_THRESHOLD = 0.15


# ── Validation ─────────────────────────────────────────────────────────────

def validate_input(text: str) -> Tuple[bool, str]:
    """Return (is_valid, error_message)."""
    stripped = text.strip()
    if not stripped:
        return False, "Please enter some text before analysing."
    if len(stripped) > MAX_INPUT_CHARS:
        return False, (
            f"Input is too long ({len(stripped)} chars). "
            f"Please keep it under {MAX_INPUT_CHARS} characters."
        )
    return True, ""


# ── Cleaning ────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase + strip URLs, @mentions, HTML tags, extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Tokenisation ────────────────────────────────────────────────────────────

def tokenize_and_lemmatize(text: str) -> List[str]:
    """Tokenise → remove stopwords (keep negations) → lemmatise."""
    tokens = word_tokenize(text)
    return [
        _LEMMATIZER.lemmatize(t)
        for t in tokens
        if t not in string.punctuation
        and (t not in _STOPWORDS or t in _KEEP_WORDS)
    ]


def preprocess_for_bilstm(text: str) -> str:
    """
    Full pipeline for BiLSTM: clean → tokenise → lemmatise → join.
    Returns a single string ready for Keras Tokenizer.texts_to_sequences().
    """
    return " ".join(tokenize_and_lemmatize(clean_text(text)))


def preprocess_for_bert(text: str) -> str:
    """
    Light cleaning for DistilBERT.
    The HuggingFace tokeniser handles sub-word splitting; we just normalise.
    """
    return clean_text(text)


# ── Convenience wrapper ─────────────────────────────────────────────────────

def full_preprocess(text: str) -> dict:
    """
    Returns dict with:
      original      : stripped raw text
      bilstm_input  : cleaned + lemmatised string
      bert_input    : lightly cleaned string
    """
    stripped = text.strip()
    return {
        "original":     stripped,
        "bilstm_input": preprocess_for_bilstm(stripped),
        "bert_input":   preprocess_for_bert(stripped),
    }
