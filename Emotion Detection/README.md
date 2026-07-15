# Emotion-Aware Learning Support Engine (EALSE)

A Streamlit web app that detects the emotional state behind student or educator text
and generates personalised, empathetic guidance using Google Gemini AI.

## Project Structure

```
Emotion/
├── data/                          # Runtime data (created automatically)
├── models/
│   ├── bert_emotion_model_final/  # Fine-tuned DistilBERT (after training)
│   └── bilstm/                    # BiLSTM .h5 + tokenizer.pkl (after training)
├── notebooks/                     # Training notebooks
├── src/
│   ├── preprocessing.py           # Text cleaning, tokenisation, lemmatisation
│   ├── model.py                   # BiLSTM architecture + GoEmotions label map
│   ├── bert_model.py              # DistilBERT architecture + dataset helper
│   ├── train.py                   # Training script
│   └── predict.py                 # Unified prediction engine
├── app.py                         # Streamlit entry point
├── emotion_response_mapping.csv   # Emotion → tone/response strategy reference
├── emotion_response_examples.csv  # Labelled test examples for QA
├── PROJECT_ANALYSIS_REPORT.md     # Architecture and design decisions
├── requirements.txt
└── .env                           # GEMINI_API_KEY (never commit this)
```

## Quick Start

### 1. Install dependencies
```bash
conda create -n ealse python=3.10 -y
conda activate ealse
pip install -r requirements.txt
```

### 2. Configure Gemini API
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Run the app (after training)
```bash
streamlit run app.py
```

### 4. Train models (auto-downloads GoEmotions dataset)
```bash
python src/train.py --model both
```
This auto-downloads GoEmotions (~400MB, cached) and saves models to `models/`.

## Emotion Classes
| Class | GoEmotions Sources |
|-------|--------------------|
| Confused | confusion, realization |
| Frustrated | anger, annoyance, disappointment, disapproval |
| Curious | curiosity, surprise |
| Confident | approval, admiration, pride |
| Excited | excitement, joy, amusement, optimism, love, desire |
| Anxious | fear, nervousness, embarrassment, caring |
| Neutral | neutral, relief, gratitude, sadness, grief, remorse, disgust |

## Tech Stack
- **ML**: TensorFlow/Keras (BiLSTM) · PyTorch + HuggingFace Transformers (DistilBERT)
- **NLP**: NLTK
- **Generative AI**: Google Gemini 1.5 Flash
- **Web app**: Streamlit · Plotly
- **Data**: GoEmotions (Kaggle)
