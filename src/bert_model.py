"""
DistilBERT model definition and dataset helper (PyTorch / HuggingFace).
Compatible with transformers 5.x and Python 3.13.
"""

from __future__ import annotations
from typing import List

# ── Constants ──────────────────────────────────────────────────────────────
BERT_BASE     = "distilbert-base-uncased"
BERT_MAX_LEN  = 64
BATCH_SIZE    = 16
EPOCHS        = 4
LEARNING_RATE = 2e-5

EMOTION_CLASSES: List[str] = [
    "Confused", "Frustrated", "Curious",
    "Confident", "Excited", "Anxious", "Neutral",
]
NUM_CLASSES = len(EMOTION_CLASSES)
LABEL2IDX   = {e: i for i, e in enumerate(EMOTION_CLASSES)}
IDX2LABEL   = {i: e for i, e in enumerate(EMOTION_CLASSES)}


# ── Model factory ──────────────────────────────────────────────────────────

def get_tokenizer():
    """Return the DistilBERT fast tokeniser (downloads on first call)."""
    try:
        from transformers import DistilBertTokenizerFast
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install with: pip install transformers"
        ) from exc
    return DistilBertTokenizerFast.from_pretrained(BERT_BASE)


def build_bert_model(num_labels: int = NUM_CLASSES):
    """
    Load distilbert-base-uncased with a fresh 7-class classification head.
    Sets id2label / label2id so save_pretrained() stores the mapping.
    """
    try:
        from transformers import DistilBertForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "transformers is required. Install with: pip install transformers"
        ) from exc

    return DistilBertForSequenceClassification.from_pretrained(
        BERT_BASE,
        num_labels=num_labels,
        id2label=IDX2LABEL,
        label2id=LABEL2IDX,
        ignore_mismatched_sizes=True,
    )


# ── PyTorch Dataset ────────────────────────────────────────────────────────

class EmotionDataset:
    """
    torch.utils.data.Dataset wrapping pre-tokenised inputs and integer labels.
    """

    def __init__(self, texts, labels, tokenizer, max_length: int = BERT_MAX_LEN):
        try:
            import torch
            from torch.utils.data import Dataset  # noqa: F401 – keep for isinstance checks
        except ImportError as exc:
            raise ImportError(
                "PyTorch is required. Install with: pip install torch"
            ) from exc

        self._torch = torch
        self.labels = labels

        self.encodings = tokenizer(
            list(texts),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = self._torch.tensor(self.labels[idx], dtype=self._torch.long)
        return item
