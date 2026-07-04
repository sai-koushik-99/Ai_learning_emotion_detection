"""
Downloads trained model files from Google Drive at startup.
Called by app.py when running on Streamlit Cloud (model files not in repo).

File IDs are read from Streamlit Secrets or environment variables.
Falls back to hardcoded public IDs if secrets are not set.
"""

import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

BILSTM_DIR = Path("models/bilstm")
BERT_DIR   = Path("models/bert_emotion_model_final")

# ── Public Google Drive file IDs (set via Streamlit Secrets to override) ──
_DEFAULT_IDS = {
    "GDRIVE_BILSTM_H5":    "1n0qW7GMaewqXeGOeX6r9utWY8oalelpU",
    "GDRIVE_BILSTM_TOK":   "1zH_5lHE_VzbMMi_nkUlY2dOYXB_OymEM",
    "GDRIVE_BERT_WEIGHTS": "1JmDVeow6zQ4d4vpwErAvk7zpnTjIkizq",
}

_FILE_PATHS = {
    "GDRIVE_BILSTM_H5":    BILSTM_DIR / "bilstm_emotion_model.h5",
    "GDRIVE_BILSTM_TOK":   BILSTM_DIR / "tokenizer.pkl",
    "GDRIVE_BERT_WEIGHTS": BERT_DIR   / "model.safetensors",
}


def _download(file_id: str, dest: Path) -> bool:
    """Download a single file from Google Drive using gdown."""
    if dest.exists():
        log.info("Already exists, skipping: %s", dest.name)
        return True
    try:
        import gdown
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        log.info("Downloading %s ...", dest.name)
        gdown.download(url, str(dest), quiet=False, fuzzy=True)
        if dest.exists() and dest.stat().st_size > 1000:
            log.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return True
        else:
            log.error("Download produced empty/missing file: %s", dest.name)
            return False
    except Exception as exc:
        log.error("Failed to download %s: %s", dest.name, exc)
        return False


def ensure_models() -> tuple[bool, str]:
    """
    Download all model files if not present locally.
    Returns (success, error_message).
    """
    all_present = all(path.exists() for path in _FILE_PATHS.values())
    if all_present:
        log.info("All model files already present.")
        return True, ""

    log.info("Downloading missing model files from Google Drive...")

    for env_var, dest in _FILE_PATHS.items():
        if dest.exists():
            continue
        # Use secret if set, else fall back to default public ID
        file_id = os.getenv(env_var, _DEFAULT_IDS[env_var])
        if not file_id:
            return False, f"No file ID for {env_var}"
        if not _download(file_id, dest):
            return False, (
                f"Failed to download {dest.name}. "
                "Check that the Google Drive file is publicly shared."
            )

    log.info("All model files ready.")
    return True, ""
