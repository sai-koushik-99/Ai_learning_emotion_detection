"""
Training script for the Emotion-Aware Learning Support Engine.

Supports training BiLSTM (TensorFlow/Keras) and/or DistilBERT (PyTorch/HuggingFace)
on the GoEmotions dataset, remapped to 7 emotion classes.

Usage
-----
# Train BiLSTM only:
    python src/train.py --model bilstm

# Train DistilBERT only:
    python src/train.py --model bert

# Train both:
    python src/train.py --model both

# Use a custom CSV dataset (columns: text, label):
    python src/train.py --model both --dataset path/to/data.csv

# Point at a local GoEmotions directory (TSV files):
    python src/train.py --model both --data_dir /path/to/goemotions

Dataset
-------
If neither --dataset nor --data_dir is provided (or the directory is empty),
the script downloads GoEmotions automatically via the HuggingFace `datasets`
library (requires internet access on first run; cached afterwards).

Outputs
-------
BiLSTM     → models/bilstm/bilstm_emotion_model.h5
              models/bilstm/tokenizer.pkl
DistilBERT → models/bert_emotion_model_final/   (HuggingFace save_pretrained format)
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Paths ──────────────────────────────────────────────────────────────────
_ROOT      = Path(__file__).resolve().parent.parent
BILSTM_DIR = _ROOT / "models" / "bilstm"
BERT_DIR   = _ROOT / "models" / "bert_emotion_model_final"

BILSTM_DIR.mkdir(parents=True, exist_ok=True)
BERT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Local imports (must run from project root) ─────────────────────────────
sys.path.insert(0, str(_ROOT))
from src.preprocessing import (
    preprocess_for_bilstm, preprocess_for_bert,
    EMOTION_CLASSES, NUM_CLASSES, LABEL2IDX,
)
from src.model import GOEMOTIONS_MAP, build_bilstm, VOCAB_SIZE, MAX_LEN
from src.bert_model import (
    build_bert_model, get_tokenizer, EmotionDataset,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, BERT_MAX_LEN,
)


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def _map_goemotions_label(raw_label: str) -> str | None:
    """Map a single GoEmotions label string to one of 7 project classes."""
    return GOEMOTIONS_MAP.get(raw_label.strip().lower())


def load_from_hf() -> pd.DataFrame:
    """
    Download GoEmotions via HuggingFace datasets and remap labels.
    Only the 'simplified' config is used (already collapsed to 28 labels).
    """
    log.info("Downloading GoEmotions from HuggingFace datasets…")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for auto-download. "
            "Install it with:  pip install datasets"
        ) from exc

    ds = load_dataset("google-research-datasets/go_emotions", "simplified")

    # The HF version provides integer label lists; get label names from features
    label_names = ds["train"].features["labels"].feature.names

    rows = []
    for split in ("train", "validation", "test"):
        for item in ds[split]:
            text = item["text"]
            # Take the first label when multiple are present
            if not item["labels"]:
                continue
            raw_label = label_names[item["labels"][0]]
            mapped = _map_goemotions_label(raw_label)
            if mapped is None:
                continue
            rows.append({"text": text, "label": mapped})

    df = pd.DataFrame(rows)
    log.info("Loaded %d samples from HuggingFace GoEmotions.", len(df))
    return df


def load_from_tsv_dir(data_dir: str) -> pd.DataFrame:
    """
    Load GoEmotions TSV files (train.tsv / dev.tsv / test.tsv) from a
    local directory.  Expects an emotions.txt index file alongside them.
    """
    data_path = Path(data_dir)
    dfs = []
    for fname in ("train.tsv", "dev.tsv", "test.tsv"):
        p = data_path / fname
        if p.exists():
            df = pd.read_csv(p, sep="\t", header=None, names=["text", "labels", "id"])
            dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No GoEmotions TSV files found in '{data_dir}'. "
            "Expected train.tsv / dev.tsv / test.tsv."
        )

    df = pd.concat(dfs, ignore_index=True)
    df["label_raw"] = df["labels"].astype(str).str.split(",").str[0].str.strip()

    emotions_file = data_path / "emotions.txt"
    if emotions_file.exists():
        with open(emotions_file) as f:
            idx_to_name = {str(i): e.strip() for i, e in enumerate(f)}
        df["label_name"] = df["label_raw"].map(idx_to_name)
    else:
        df["label_name"] = df["label_raw"]

    df["label"] = df["label_name"].apply(_map_goemotions_label)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    log.info("Loaded %d samples from local TSV directory.", len(df))
    return df[["text", "label"]]


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """
    Load a custom CSV dataset.

    Expected format — at minimum two columns:
      text   : the input text
      label  : one of the 7 emotion class names (case-insensitive)

    Unknown labels are dropped with a warning.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: '{csv_path}'")

    df = pd.read_csv(path)

    # Accept common alternative column names
    col_map = {}
    for col in df.columns:
        if col.lower() in ("text", "sentence", "utterance", "input"):
            col_map[col] = "text"
        elif col.lower() in ("label", "emotion", "class", "target"):
            col_map[col] = "label"
    df = df.rename(columns=col_map)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "CSV must contain 'text' and 'label' columns "
            f"(found: {list(df.columns)})"
        )

    # Normalise label capitalisation
    valid = {e.lower(): e for e in EMOTION_CLASSES}
    df["label"] = df["label"].str.strip().str.lower().map(valid)
    before = len(df)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        log.warning("Dropped %d rows with unrecognised labels.", dropped)

    log.info("Loaded %d samples from CSV '%s'.", len(df), csv_path)
    return df[["text", "label"]]


def prepare_dataframe(args) -> pd.DataFrame:
    """Route to the correct loader based on CLI arguments."""
    if args.dataset:
        df = load_from_csv(args.dataset)
    elif args.data_dir:
        tsv_files = list(Path(args.data_dir).glob("*.tsv"))
        if tsv_files:
            df = load_from_tsv_dir(args.data_dir)
        else:
            csvs = list(Path(args.data_dir).glob("*.csv"))
            if csvs:
                df = load_from_csv(str(csvs[0]))
            else:
                raise FileNotFoundError(
                    f"No TSV or CSV files found in data_dir='{args.data_dir}'"
                )
    else:
        # Default: auto-download via HuggingFace
        df = load_from_hf()

    # --fast mode: stratified sample per class for quick smoke-test
    if getattr(args, "fast", False):
        n = getattr(args, "fast_samples", 3000)
        per_class = max(1, n // df["label"].nunique())
        sampled = [
            g.sample(min(len(g), per_class), random_state=42)
            for _, g in df.groupby("label")
        ]
        df = pd.concat(sampled, ignore_index=True)
        log.info("--fast mode: using %d samples (stratified)", len(df))

    return df


# ══════════════════════════════════════════════════════════════════════════
# BiLSTM TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_bilstm(df: pd.DataFrame) -> None:
    """Train the BiLSTM model and save artefacts to models/bilstm/."""
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for BiLSTM training. "
            "Install with:  pip install tensorflow"
        ) from exc

    log.info("─── BiLSTM Training ──────────────────────────────────")
    log.info("Preprocessing %d samples…", len(df))

    df = df.copy()
    df["clean"]     = df["text"].apply(preprocess_for_bilstm)
    df["label_idx"] = df["label"].map(LABEL2IDX)

    X_train, X_val, y_train, y_val = train_test_split(
        df["clean"].tolist(),
        df["label_idx"].tolist(),
        test_size=0.15,
        random_state=42,
        stratify=df["label_idx"],
    )

    # Build Keras tokeniser on training text only
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_tr = pad_sequences(
        tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding="post", truncating="post"
    )
    X_v = pad_sequences(
        tokenizer.texts_to_sequences(X_val), maxlen=MAX_LEN, padding="post", truncating="post"
    )
    y_tr = np.array(y_train)
    y_v  = np.array(y_val)

    # Class weights to handle imbalanced GoEmotions distribution
    classes   = np.unique(y_tr)
    cw_values = compute_class_weight("balanced", classes=classes, y=y_tr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw_values)}

    model = build_bilstm(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
    model.summary(print_fn=log.info)

    best_h5 = str(BILSTM_DIR / "best_checkpoint.h5")
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_h5, monitor="val_accuracy", save_best_only=True, verbose=0),
    ]

    log.info("Training BiLSTM (up to 20 epochs, early-stop patience=3)…")
    model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v),
        epochs=20,
        batch_size=64,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────
    log.info("Evaluating on validation set…")
    y_pred = model.predict(X_v, verbose=0).argmax(axis=1)
    report = classification_report(
        y_v, y_pred,
        target_names=EMOTION_CLASSES,
        digits=4,
    )
    log.info("\n%s", report)

    # ── Save artefacts ────────────────────────────────────────────────────
    final_h5 = str(BILSTM_DIR / "bilstm_emotion_model.h5")
    tok_pkl  = str(BILSTM_DIR / "tokenizer.pkl")

    model.save(final_h5)
    with open(tok_pkl, "wb") as f:
        pickle.dump(tokenizer, f)

    log.info("BiLSTM model  → %s", final_h5)
    log.info("Tokenizer     → %s", tok_pkl)
    log.info("─── BiLSTM done ──────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════
# DistilBERT TRAINING
# ══════════════════════════════════════════════════════════════════════════

def train_bert(df: pd.DataFrame, n_epochs: int = EPOCHS) -> None:
    """Fine-tune DistilBERT and save artefacts to models/bert_emotion_model_final/."""
    try:
        import torch
        from torch.utils.data import DataLoader
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError(
            "PyTorch and transformers are required for DistilBERT training. "
            "Install with:  pip install torch transformers"
        ) from exc

    log.info("─── DistilBERT Training ──────────────────────────────")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Device: %s  |  Epochs: %d", device, n_epochs)

    df = df.copy()
    df["clean"]     = df["text"].apply(preprocess_for_bert)
    df["label_idx"] = df["label"].map(LABEL2IDX)

    X_train, X_val, y_train, y_val = train_test_split(
        df["clean"].tolist(),
        df["label_idx"].tolist(),
        test_size=0.15,
        random_state=42,
        stratify=df["label_idx"],
    )

    log.info("Loading DistilBERT tokeniser…")
    tokenizer = get_tokenizer()

    train_ds = EmotionDataset(X_train, y_train, tokenizer, BERT_MAX_LEN)
    val_ds   = EmotionDataset(X_val,   y_val,   tokenizer, BERT_MAX_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    log.info("Loading DistilBERT base model…")
    model = build_bert_model()
    model.to(device)

    optimizer   = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_dl) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    best_val_acc = 0.0

    for epoch in range(1, n_epochs + 1):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            if step % 100 == 0:
                log.info(
                    "  Epoch %d/%d  step %d/%d  avg_loss=%.4f",
                    epoch, EPOCHS, step, len(train_dl), total_loss / step,
                )

        avg_loss = total_loss / len(train_dl)

        # ── Validate ────────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                batch  = {k: v.to(device) for k, v in batch.items()}
                preds  = model(**batch).logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())

        val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
        log.info(
            "Epoch %d/%d  avg_loss=%.4f  val_acc=%.4f",
            epoch, n_epochs, avg_loss, val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(str(BERT_DIR))
            tokenizer.save_pretrained(str(BERT_DIR))
            log.info("  ✓ Best model checkpoint saved (val_acc=%.4f)", best_val_acc)

    # ── Final evaluation report ────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        target_names=EMOTION_CLASSES,
        digits=4,
    )
    log.info("Final validation report:\n%s", report)
    log.info("DistilBERT saved → %s", BERT_DIR)
    log.info("─── DistilBERT done ──────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train emotion classification models for EALSE.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--model",
        choices=["bilstm", "bert", "both"],
        default="both",
        help=(
            "Which model(s) to train:\n"
            "  bilstm  — BiLSTM (TensorFlow/Keras)\n"
            "  bert    — DistilBERT (PyTorch/HuggingFace)\n"
            "  both    — train both sequentially (default)"
        ),
    )
    p.add_argument(
        "--dataset",
        default=None,
        metavar="CSV_PATH",
        help=(
            "Path to a custom CSV file with 'text' and 'label' columns.\n"
            "Labels must be one of: " + ", ".join(EMOTION_CLASSES)
        ),
    )
    p.add_argument(
        "--data_dir",
        default=None,
        metavar="DIR",
        help=(
            "Path to a local GoEmotions directory containing\n"
            "train.tsv / dev.tsv / test.tsv (+ emotions.txt).\n"
            "If omitted, GoEmotions is downloaded automatically."
        ),
    )
    p.add_argument(
        "--fast",
        action="store_true",
        default=False,
        help=(
            "Quick smoke-test mode: train on a small stratified subset\n"
            "(~3000 samples) so you can verify the pipeline end-to-end\n"
            "in minutes rather than hours. Use for testing only."
        ),
    )
    p.add_argument(
        "--fast_samples",
        type=int,
        default=3000,
        metavar="N",
        help="Number of samples to use in --fast mode (default: 3000).",
    )
    p.add_argument(
        "--bert_epochs",
        type=int,
        default=None,
        metavar="N",
        help="Override BERT training epochs (default: 4 from bert_model.py).",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    log.info("═══════════════════════════════════════════════════")
    log.info(" Emotion-Aware Learning Support Engine — Training  ")
    log.info("═══════════════════════════════════════════════════")
    log.info("Target model(s): %s", args.model)

    df = prepare_dataframe(args)

    # Show class distribution
    dist = df["label"].value_counts().to_dict()
    log.info("Class distribution: %s", dist)

    if args.model in ("bilstm", "both"):
        train_bilstm(df)

    if args.model in ("bert", "both"):
        n_epochs = args.bert_epochs if args.bert_epochs else EPOCHS
        train_bert(df, n_epochs=n_epochs)

    log.info("All training complete.")
