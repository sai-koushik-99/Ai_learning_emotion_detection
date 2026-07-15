"""
Unified real-model inference engine for the Emotion-Aware Learning Support Engine.

NO stub mode.  If model files are missing, ModelNotReadyError is raised with
a clear message telling the user to run training first.

Classes
-------
ModelNotReadyError   : raised when a model directory / file is absent
EmotionPredictor     : loads both models, exposes .predict(text) -> dict

Prediction result schema
------------------------
{
    "ok":                 bool,
    "error":              str | None,
    "input_text":         str,
    "bilstm": {
        "primary":            str,
        "primary_confidence": float,
        "is_mixed":           bool,
        "secondary":          str | None,
        "secondary_confidence": float | None,
        "all_scores":         dict[str, float],
    },
    "bert":   { ...same shape... },
    "final":  { ...same shape... },
}
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

from src.preprocessing import (
    validate_input,
    full_preprocess,
    EMOTION_CLASSES,
    NUM_CLASSES,
    MIXED_EMOTION_THRESHOLD,
)
from src.model import MAX_LEN as BILSTM_MAX_LEN
from src.bert_model import BERT_MAX_LEN

log = logging.getLogger(__name__)


# ── Pure-Python tokenizer (no Keras dependency) ────────────────────────────

class JsonTokenizer:
    """
    Minimal tokenizer loaded from a plain JSON file.
    Replicates Keras Tokenizer.texts_to_sequences() with no framework imports.
    """
    def __init__(self, json_path: str):
        import json
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        self._word_index = data["word_index"]
        self._oov        = data.get("oov_token", "<OOV>")
        self._oov_id     = self._word_index.get(self._oov, 1)
        self._num_words  = data.get("num_words", 10000)

    def texts_to_sequences(self, texts):
        result = []
        for text in texts:
            seq = []
            for word in str(text).lower().split():
                idx = self._word_index.get(word, self._oov_id)
                if idx < self._num_words:
                    seq.append(idx)
                else:
                    seq.append(self._oov_id)
            result.append(seq)
        return result

# ── Resolved model paths ───────────────────────────────────────────────────
_ROOT          = Path(__file__).resolve().parent.parent
BILSTM_NPZ     = _ROOT / "models" / "bilstm" / "bilstm_weights.npz"
BILSTM_TOK_JSON = _ROOT / "models" / "bilstm" / "tokenizer.json"
BILSTM_TOK_PKL = _ROOT / "models" / "bilstm" / "tokenizer.pkl"   # legacy fallback
BERT_DIR       = _ROOT / "models" / "bert_emotion_model_final"


# ══════════════════════════════════════════════════════════════════════════
# Custom exception
# ══════════════════════════════════════════════════════════════════════════

class ModelNotReadyError(RuntimeError):
    """
    Raised when a required model file or directory is missing.
    The message contains actionable instructions for the user.
    """


# ══════════════════════════════════════════════════════════════════════════
# Framework imports (fail loudly if packages not installed)
# ══════════════════════════════════════════════════════════════════════════

def _require_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        return tf, pad_sequences
    except ImportError:
        raise ModelNotReadyError(
            "TensorFlow is not installed. "
            "Run:  pip install tensorflow"
        )


def _require_torch():
    try:
        import torch
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        return torch, DistilBertTokenizerFast, DistilBertForSequenceClassification
    except ImportError:
        raise ModelNotReadyError(
            "PyTorch or transformers is not installed. "
            "Run:  pip install torch transformers"
        )


# ══════════════════════════════════════════════════════════════════════════
# Model loaders
# ══════════════════════════════════════════════════════════════════════════

def _load_bilstm():
    """
    Rebuild BiLSTM from src/model.py and load weights from a numpy .npz file.
    Fully TF/Keras version independent — no config serialization used.
    """
    tf, pad_sequences = _require_tensorflow()

    if not BILSTM_NPZ.exists():
        raise ModelNotReadyError(
            f"BiLSTM weights not found at:\n  {BILSTM_NPZ}\n\n"
            "Train it first with:\n"
            "  python src/train.py --model bilstm"
        )
    if not BILSTM_TOK_JSON.exists() and not BILSTM_TOK_PKL.exists():
        raise ModelNotReadyError(
            f"BiLSTM tokenizer not found.\n"
            "Re-run training:  python src/train.py --model bilstm"
        )

    import numpy as np
    from src.model import build_bilstm, VOCAB_SIZE, MAX_LEN, NUM_CLASSES

    log.info("Building BiLSTM and loading weights from %s …", BILSTM_NPZ)
    model = build_bilstm(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
    dummy = np.zeros((1, MAX_LEN), dtype=np.int32)
    _ = model(dummy)  # build layers

    npz = np.load(str(BILSTM_NPZ))
    weights = [npz[f"arr_{i}"] for i in range(len(npz.files))]
    model.set_weights(weights)

    # Load tokenizer — prefer JSON (no Keras dependency), fall back to pkl
    if BILSTM_TOK_JSON.exists():
        log.info("Loading tokenizer from JSON...")
        tokenizer = JsonTokenizer(str(BILSTM_TOK_JSON))
    elif BILSTM_TOK_PKL.exists():
        log.info("Loading tokenizer from pkl...")
        with open(BILSTM_TOK_PKL, "rb") as f:
            tokenizer = pickle.load(f)
    else:
        raise ModelNotReadyError(
            f"BiLSTM tokenizer not found.\n"
            "Re-run training:  python src/train.py --model bilstm"
        )
    log.info("BiLSTM loaded.")
    return model, tokenizer, pad_sequences


def _load_bert():
    """Load DistilBERT model and tokeniser. Raises ModelNotReadyError if missing."""
    torch, TokenizerFast, ModelClass = _require_torch()

    config_file = BERT_DIR / "config.json"
    if not config_file.exists():
        raise ModelNotReadyError(
            f"DistilBERT model not found at:\n  {BERT_DIR}\n\n"
            "Train it first with:\n"
            "  python src/train.py --model bert"
        )

    log.info("Loading DistilBERT from %s …", BERT_DIR)
    tokenizer = TokenizerFast.from_pretrained(str(BERT_DIR))
    model     = ModelClass.from_pretrained(str(BERT_DIR))
    model.eval()
    log.info("DistilBERT loaded.")
    return model, tokenizer, torch


# ══════════════════════════════════════════════════════════════════════════
# Score → result conversion
# ══════════════════════════════════════════════════════════════════════════

def _scores_to_result(scores: np.ndarray) -> dict:
    """Convert a softmax probability vector to the canonical result dict."""
    idx = np.argsort(scores)[::-1]
    pri_idx = int(idx[0])
    sec_idx = int(idx[1])
    sec_conf = float(scores[sec_idx])
    is_mixed = sec_conf >= MIXED_EMOTION_THRESHOLD

    return {
        "primary":              EMOTION_CLASSES[pri_idx],
        "primary_confidence":   float(scores[pri_idx]),
        "is_mixed":             is_mixed,
        "secondary":            EMOTION_CLASSES[sec_idx] if is_mixed else None,
        "secondary_confidence": sec_conf if is_mixed else None,
        "all_scores": {
            EMOTION_CLASSES[i]: float(scores[i]) for i in range(NUM_CLASSES)
        },
    }


def _compute_final(bert: dict, bilstm: dict) -> dict:
    """
    DistilBERT is the primary model.
    When the top-two BERT probabilities are within 2pp, blend 50/50 with BiLSTM.
    """
    b = np.array([bert["all_scores"][e]   for e in EMOTION_CLASSES])
    l = np.array([bilstm["all_scores"][e] for e in EMOTION_CLASSES])

    sorted_b = np.argsort(b)[::-1]
    gap = b[sorted_b[0]] - b[sorted_b[1]]

    if gap < 0.02:
        blended = (b + l) / 2.0
        blended /= blended.sum()
        scores = blended
    else:
        scores = b

    return _scores_to_result(scores)


# ══════════════════════════════════════════════════════════════════════════
# EmotionPredictor
# ══════════════════════════════════════════════════════════════════════════

class EmotionPredictor:
    """
    Loads both trained classifiers at construction time.

    Raises ModelNotReadyError immediately if any model file is missing,
    so the Streamlit app can catch it and display setup instructions.

    Cache with st.cache_resource so models load once per session.
    """

    def __init__(self):
        # BiLSTM
        self._bilstm_model, self._bilstm_tok, self._pad = _load_bilstm()

        # DistilBERT
        self._bert_model, self._bert_tok, self._torch = _load_bert()

        # Move BERT to GPU if available
        self._device = "cpu"
        if self._torch.cuda.is_available():
            self._device = "cuda"
            self._bert_model.to(self._device)
            log.info("DistilBERT moved to CUDA.")

    # ── Public ─────────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Run full inference on a single text string.

        Returns the canonical result dict described in the module docstring.
        Returns {"ok": False, "error": <message>} on validation failure.
        On unexpected inference errors, re-raises rather than silently failing.
        """
        valid, err = validate_input(text)
        if not valid:
            return {"ok": False, "error": err}

        pp = full_preprocess(text)

        bilstm_result = self._run_bilstm(pp["bilstm_input"])
        bert_result   = self._run_bert(pp["bert_input"])
        final         = _compute_final(bert_result, bilstm_result)

        return {
            "ok":         True,
            "error":      None,
            "input_text": text.strip(),
            "bilstm":     bilstm_result,
            "bert":       bert_result,
            "final":      final,
        }

    # ── Internal inference ─────────────────────────────────────────────────

    def _run_bilstm(self, bilstm_input: str) -> dict:
        seq    = self._bilstm_tok.texts_to_sequences([bilstm_input])
        padded = self._pad(
            seq, maxlen=BILSTM_MAX_LEN, padding="post", truncating="post"
        )
        probs = self._bilstm_model.predict(padded, verbose=0)[0]
        return _scores_to_result(probs)

    def _run_bert(self, bert_input: str) -> dict:
        enc = self._bert_tok(
            bert_input,
            return_tensors="pt",
            max_length=BERT_MAX_LEN,
            truncation=True,
            padding="max_length",
        )
        enc = {k: v.to(self._device) for k, v in enc.items()}
        with self._torch.no_grad():
            logits = self._bert_model(**enc).logits.squeeze(0)
            probs  = self._torch.softmax(logits, dim=-1).cpu().numpy()
        return _scores_to_result(probs)
