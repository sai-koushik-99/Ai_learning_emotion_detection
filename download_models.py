"""
Downloads trained model files from Google Drive at startup.
Called by app.py when running on Streamlit Cloud (model files not in repo).

Set these in Streamlit Cloud Secrets (or .env locally):
  GDRIVE_BILSTM_H5   = "your_file_id"
  GDRIVE_BILSTM_TOK  = "your_file_id"
  GDRIVE_BERT_WEIGHTS = "your_file_id"
"""

import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

BILSTM_DIR = Path("models/bilstm")
BERT_DIR   = Path("models/bert_emotion_model_final")


def _download(file_id: str, dest: Path) -> bool:
    """Download a single file from Google Drive using gdown."""
    if dest.exists():
        log.info("Already exists, skipping: %s", dest)
        return True
    try:
        import gdown
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        log.info("Downloading %s ...", dest.name)
        gdown.download(url, str(dest), quiet=False)
        return dest.exists()
    except Exception as exc:
        log.error("Failed to download %s: %s", dest.name, exc)
        return False


def ensure_models() -> tuple[bool, str]:
    """
    Download all model files if not present.
    Returns (success, error_message).
    """
    ids = {
        "GDRIVE_BILSTM_H5":    BILSTM_DIR / "bilstm_emotion_model.h5",
        "GDRIVE_BILSTM_TOK":   BILSTM_DIR / "tokenizer.pkl",
        "GDRIVE_BERT_WEIGHTS": BERT_DIR   / "model.safetensors",
    }

    # Check if all files already present
    all_present = all(path.exists() for path in ids.values())
    if all_present:
        return True, ""

    # Try to download missing files
    missing_ids = []
    for env_var, dest in ids.items():
        if dest.exists():
            continue
        file_id = os.getenv(env_var, "")
        if not file_id:
            missing_ids.append(env_var)
            continue
        if not _download(file_id, dest):
            return False, f"Failed to download {dest.name}. Check {env_var} in secrets."

    if missing_ids:
        return False, (
            "Model download IDs not configured. "
            f"Set these secrets: {', '.join(missing_ids)}"
        )

    return True, ""
