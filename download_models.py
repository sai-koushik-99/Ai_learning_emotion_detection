"""
Downloads trained model files from Google Drive at startup.
Uses gdown with fuzzy=True to handle Google's virus-scan redirect for large files.
"""

import os
import logging
from pathlib import Path

log = logging.getLogger(__name__)

BILSTM_DIR = Path("models/bilstm")
BERT_DIR   = Path("models/bert_emotion_model_final")

# Google Drive file IDs
_FILE_MAP = {
    "GDRIVE_BILSTM_KERAS": (BILSTM_DIR / "bilstm_emotion_model.keras", "18JKt9ba3eGpAZKxYoEBWv3XKLKBZeYXy"),
    "GDRIVE_BILSTM_TOK":   (BILSTM_DIR / "tokenizer.pkl",              "1zH_5lHE_VzbMMi_nkUlY2dOYXB_OymEM"),
    "GDRIVE_BERT_WEIGHTS": (BERT_DIR   / "model.safetensors",          "1JmDVeow6zQ4d4vpwErAvk7zpnTjIkizq"),
}


def _download_file(file_id: str, dest: Path) -> bool:
    """Download from Google Drive with multiple fallback methods."""
    if dest.exists() and dest.stat().st_size > 1000:
        log.info("Already exists: %s", dest.name)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Method 1: gdown (without fuzzy for compatibility with older versions)
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}&export=download"
        log.info("Downloading %s via gdown...", dest.name)
        gdown.download(url, str(dest), quiet=False)
        if dest.exists() and dest.stat().st_size > 1000:
            log.info("Downloaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return True
        log.warning("gdown produced empty file for %s, trying fallback...", dest.name)
    except Exception as e:
        log.warning("gdown failed for %s: %s", dest.name, e)

    # Method 2: requests with session (handles cookies for large files)
    try:
        import requests
        log.info("Downloading %s via requests...", dest.name)
        session = requests.Session()
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = session.get(url, stream=True, timeout=300)

        # Handle Google's virus-scan confirmation for large files
        token = None
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                token = value
                break

        if token:
            url = f"https://drive.google.com/uc?export=download&confirm={token}&id={file_id}"
            response = session.get(url, stream=True, timeout=300)

        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)

        if dest.exists() and dest.stat().st_size > 1000:
            log.info("Downloaded %s via requests (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
            return True
    except Exception as e:
        log.warning("requests fallback failed for %s: %s", dest.name, e)

    log.error("All download methods failed for %s", dest.name)
    return False


def ensure_models() -> tuple[bool, str]:
    """
    Download all model files if not present.
    Returns (success, error_message).
    """
    all_present = all(
        path.exists() and path.stat().st_size > 1000
        for path, _ in _FILE_MAP.values()
    )
    if all_present:
        log.info("All model files already present.")
        return True, ""

    log.info("Downloading missing model files from Google Drive...")

    for env_var, (dest, default_id) in _FILE_MAP.items():
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        file_id = os.getenv(env_var, default_id)
        if not _download_file(file_id, dest):
            return False, (
                f"Failed to download {dest.name}. "
                "Make sure the Google Drive file is shared as 'Anyone with the link can view'."
            )

    log.info("All model files ready.")
    return True, ""
