"""
Saves BiLSTM weights as a plain numpy .npz file.
100% framework-version-independent — works on any TF/Keras version.
Run: python resave_bilstm.py
"""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, ".")

import numpy as np
import tensorflow as tf
print(f"TF: {tf.__version__}")

from src.model import build_bilstm, VOCAB_SIZE, MAX_LEN, NUM_CLASSES

H5_SRC   = "models/bilstm/bilstm_emotion_model.h5"
NPZ_OUT  = "models/bilstm/bilstm_weights.npz"

print("Loading original model...")
old = tf.keras.models.load_model(H5_SRC)
weights = old.get_weights()
print(f"  {len(weights)} weight arrays")

print("Saving as numpy .npz ...")
np.savez(NPZ_OUT, *weights)
size = os.path.getsize(NPZ_OUT) / 1e6
print(f"  Saved: {size:.1f} MB")

# Verify round-trip
print("Verifying round-trip...")
new = build_bilstm(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
dummy = np.zeros((1, MAX_LEN), dtype=np.int32)
_ = new(dummy)
loaded = np.load(NPZ_OUT)
new.set_weights([loaded[f"arr_{i}"] for i in range(len(weights))])
diff = np.abs(old.predict(dummy, verbose=0) - new.predict(dummy, verbose=0)).max()
print(f"  Max diff: {diff:.2e}  (must be 0)")
print()
print("Upload models/bilstm/bilstm_weights.npz to Google Drive and share the link.")
