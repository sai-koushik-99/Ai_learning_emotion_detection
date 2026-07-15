"""
BiLSTM model definition (TensorFlow / Keras).

Architecture:
  Embedding(vocab_size, 128, input_length=50)
  → Bidirectional LSTM(64, return_sequences=True)
  → Bidirectional LSTM(32)
  → Dense(64, relu) → Dropout(0.3) → Dense(7, softmax)

Also defines the GoEmotions-28 → 7-class label mapping used by train.py.
"""

from __future__ import annotations

# ── GoEmotions 28-label → 7-class mapping ─────────────────────────────────
GOEMOTIONS_MAP: dict[str, str] = {
    # Confused
    "confusion":      "Confused",
    "realization":    "Confused",
    # Frustrated
    "anger":          "Frustrated",
    "annoyance":      "Frustrated",
    "disappointment": "Frustrated",
    "disapproval":    "Frustrated",
    # Curious
    "curiosity":      "Curious",
    "surprise":       "Curious",
    # Confident
    "approval":       "Confident",
    "admiration":     "Confident",
    "pride":          "Confident",
    # Excited
    "excitement":     "Excited",
    "joy":            "Excited",
    "amusement":      "Excited",
    "optimism":       "Excited",
    "love":           "Excited",
    "desire":         "Excited",
    # Anxious
    "fear":           "Anxious",
    "nervousness":    "Anxious",
    "embarrassment":  "Anxious",
    "caring":         "Anxious",
    # Neutral
    "neutral":        "Neutral",
    "relief":         "Neutral",
    "gratitude":      "Neutral",
    "sadness":        "Neutral",
    "grief":          "Neutral",
    "remorse":        "Neutral",
    "disgust":        "Neutral",
}

# ── Hyperparameters ────────────────────────────────────────────────────────
VOCAB_SIZE  = 10_000
EMBED_DIM   = 128
MAX_LEN     = 50
NUM_CLASSES = 7


def build_bilstm(
    vocab_size: int = VOCAB_SIZE,
    embed_dim: int  = EMBED_DIM,
    max_len: int    = MAX_LEN,
    num_classes: int = NUM_CLASSES,
):
    """
    Build and compile the BiLSTM Keras model.
    Requires TensorFlow to be installed.
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build the BiLSTM model. "
            "Install it with: pip install tensorflow"
        ) from exc

    model = keras.Sequential(
        [
            layers.Embedding(vocab_size, embed_dim, input_length=max_len),
            layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ],
        name="bilstm_emotion",
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
