# Project Analysis Report — EALSE

## 1. Architecture Overview

```
User Input (Streamlit)
       │
       ▼
 Input Validation  ──▶  Error banner if empty / >2000 chars
       │
       ▼
 Text Preprocessing  (src/preprocessing.py)
  • Lowercase, URL/mention/HTML strip
  • NLTK tokenise + lemmatise + stopword filter (negations kept)
  • Keyword-prior vector (NUM_CLASSES=7)
  • Separate outputs: bilstm_input, bert_input
       │
       ├──────────────────────┐
       ▼                      ▼
 BiLSTM Classifier      DistilBERT Classifier
 (src/model.py          (src/bert_model.py
  src/predict.py)        src/predict.py)
  TF/Keras .h5           PyTorch HF model dir
  Stub if missing        Stub if missing
       │                      │
       └──────────┬───────────┘
                  ▼
         Keyword Adjustment
         (10% prior blend)
                  │
                  ▼
         Final Prediction
         (DistilBERT primary;
          BiLSTM tiebreaker if gap < 2%)
                  │
                  ▼
         Mixed Emotion Detection
         (secondary if score ≥ 15%)
                  │
                  ▼
         Gemini Guidance Engine  ──▶  Fallback stub if no API key
         (role + subject + emotion context)
                  │
                  ▼
         Streamlit Display
         • Model comparison cards
         • Final prediction summary
         • Guidance + Regenerate
         • Analytics dashboard
                  │
                  ▼
         CSV Logger  →  data/interactions.csv
```

## 2. Design Decisions

### 2.1 DistilBERT over full BERT
DistilBERT is 40% smaller and 60% faster than BERT-base while retaining ~97% of
performance. This meets the <5s end-to-end latency target on Intel i3 / 4GB RAM.

### 2.2 Stub mode
Both classifiers degrade gracefully to keyword-nudged uniform distributions when
model files are missing. This allows the full UI to be built, tested, and demoed
before Kaggle training completes — no blocking dependency on trained weights.

### 2.3 Keyword-prior adjustment (10% blend weight)
Strong keyword signals (e.g., "stuck", "confused", "can't figure out") provide a
soft prior that nudges — but never overrides — model softmax scores.
The 10% weight keeps the model dominant while reducing the chance of a correct
keyword signal being buried by irrelevant context.

### 2.4 Mixed-emotion threshold (≥ 15%)
A 15% secondary-emotion confidence threshold was chosen to flag genuine compound
states (e.g., Curious + Confused at 18%) while avoiding noise from softmax
probability spreading across all 7 classes at low confidence levels.

### 2.5 Final prediction tie-breaking
When DistilBERT's top-two class probabilities are within 2 percentage points, the
BiLSTM is used as a tiebreaker via a 50/50 blend. Outside this band, DistilBERT
scores are used directly — it is the higher-quality model.

### 2.6 Regeneration temperature
Base guidance uses temperature=0.7 (focused, consistent).
Regeneration uses temperature=0.9 (more varied phrasing) without re-running the
emotion pipeline — emotion cards do not change on regeneration.

## 3. Dataset

- **Source**: GoEmotions (Google, 58k+ Reddit comments, 28 emotion labels)
- **Remapping**: 27 of 28 labels mapped to 7 project classes; "neutral" maps to Neutral
- **Class imbalance**: handled via `compute_class_weight("balanced")` for BiLSTM
  and class-weighted loss for DistilBERT
- **Split**: 85% train / 15% validation (stratified)

## 4. Performance Targets

| Metric | Target |
|--------|--------|
| Classification accuracy | ≥ 80% on held-out test set |
| End-to-end latency | < 5 seconds (i3, 4GB RAM) |
| Mixed-emotion detection | Correctly flags dual-emotion test cases |
| Memory footprint | DistilBERT ≈ 250MB, BiLSTM < 50MB |

## 5. Security Notes

- `GEMINI_API_KEY` stored in `.env` only, excluded from version control via `.gitignore`
- No PII transmitted externally beyond the student message sent to Gemini API
- CSV logs are local-only; no cloud persistence

## 6. Open Items

- [ ] Confirm final emotion class count (7 used throughout; brief mentioned 5)
- [ ] Domain adaptation: collect 200-300 real student messages for fine-tuning after base training
- [ ] Accessibility: add `aria-label` attributes to Streamlit custom HTML components
- [ ] Multi-language support: preprocessing currently English-only
