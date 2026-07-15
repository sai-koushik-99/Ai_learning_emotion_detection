"""
Re-saves the Keras tokenizer as plain JSON.
JSON has zero framework dependency - loads on any TF/Keras version.
Run: python resave_tokenizer.py
"""
import pickle, json, sys
sys.path.insert(0, ".")

TOK_PKL  = "models/bilstm/tokenizer.pkl"
TOK_JSON = "models/bilstm/tokenizer.json"

print("Loading tokenizer.pkl ...")
with open(TOK_PKL, "rb") as f:
    tok = pickle.load(f)

print(f"  Type: {type(tok)}")
print(f"  Vocab size: {len(tok.word_index)}")

# Extract everything we need to reconstruct it
tok_data = {
    "word_index":    tok.word_index,
    "index_word":    {str(k): v for k, v in tok.index_word.items()},
    "word_counts":   tok.word_counts,
    "num_words":     tok.num_words,
    "oov_token":     tok.oov_token,
    "lower":         tok.lower,
}

with open(TOK_JSON, "w", encoding="utf-8") as f:
    json.dump(tok_data, f, ensure_ascii=False)

import os
size_kb = os.path.getsize(TOK_JSON) / 1024
print(f"Saved {TOK_JSON} ({size_kb:.0f} KB)")

# Verify round-trip
print("Verifying round-trip ...")
with open(TOK_JSON) as f:
    data = json.load(f)

# Reconstruct minimal tokenizer behaviour
word_index = data["word_index"]
test_text  = "i don't understand this concept"
tokens = [word_index.get(w, word_index.get(data["oov_token"], 1))
          for w in test_text.lower().split()]
print(f"  Test encoding: {tokens}")
print("Done — upload models/bilstm/tokenizer.json to Google Drive and share the link.")
