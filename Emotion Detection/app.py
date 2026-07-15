"""
Emotion-Aware Learning Support Engine — Streamlit UI
Run:  streamlit run app.py
"""

import sys
import os
import logging
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.predict import EmotionPredictor, ModelNotReadyError
from src.preprocessing import EMOTION_CLASSES

logging.basicConfig(level=logging.INFO)

# ── Constants ──────────────────────────────────────────────────────────────
SUBJECT_OPTIONS = [
    "General / Other", "Mathematics", "Physics", "Chemistry",
    "Biology", "Computer Science", "History", "Literature",
    "Economics", "Psychology",
]
LOG_FILE = Path("data") / "interactions.csv"
LOG_FIELDS = [
    "timestamp", "input_text", "role", "subject",
    "bilstm_primary", "bilstm_conf", "bilstm_mixed", "bilstm_secondary",
    "bert_primary",   "bert_conf",   "bert_mixed",   "bert_secondary",
    "final_primary",  "final_conf",  "final_mixed",  "final_secondary",
    "gemini_response",
]
EMOTION_COLORS = {
    "Confused":   "#636EFA", "Frustrated": "#EF553B", "Curious":   "#00CC96",
    "Confident":  "#19D3F3", "Excited":    "#FF6692", "Anxious":   "#FFA15A",
    "Neutral":    "#B6E880",
}
EMOTION_EMOJI = {
    "Confused": "😕", "Frustrated": "😤", "Curious":    "🤔",
    "Confident": "😎", "Excited":   "🤩", "Anxious":   "😰",
    "Neutral":   "😐",
}

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Emotion-Aware Learning Support",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
body, .stApp { background:#0F0F1A; color:#E0E0E0; }
.title    { font-size:2rem; font-weight:700; color:#FFFFFF; }
.subtitle { color:#9999BB; margin-bottom:1.5rem; font-size:1rem; }
.card     { background:#1A1A2E; border-radius:12px; padding:1.2rem 1.4rem;
            border:1px solid #2E2E4E; margin-bottom:.8rem; }
.badge    { display:inline-block; border-radius:6px; padding:2px 9px;
            font-size:.76rem; font-weight:600; margin-left:4px; }
.badge-ok    { background:#1F8A5A; color:#fff; }
.badge-mixed { background:#636EFA; color:#fff; }
.guidance-box { background:#12122A; border-left:4px solid #636EFA;
                border-radius:8px; padding:1.2rem 1.4rem; margin-top:.5rem; }
.setup-box  { background:#1A1A2E; border:1px solid #E05A2B; border-radius:12px;
              padding:1.5rem 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading emotion models…")
def load_predictor():
    """Download models if needed, then load. Returns (predictor, None) or (None, error_message)."""

    # ── Step 1: ensure model files exist (downloads from Google Drive if missing) ──
    try:
        from download_models import ensure_models
        dl_ok, dl_err = ensure_models()
        if not dl_ok:
            return None, f"Model download failed: {dl_err}"
    except Exception as exc:
        return None, f"download_models.py error: {exc}"

    # ── Step 2: load models ────────────────────────────────────────────────
    try:
        return EmotionPredictor(), None
    except ModelNotReadyError as exc:
        return None, str(exc)
    except Exception as exc:
        return None, f"Unexpected error loading models: {exc}"


def _get_api_key() -> str:
    """Always re-reads .env so a key change takes effect without restarting.
    Explicitly uses GEMINI_API_KEY and ignores any GOOGLE_API_KEY in the env."""
    load_dotenv(override=True)
    return os.getenv("GEMINI_API_KEY", "").strip()


def _call_gemini(prompt: str, temperature: float = 0.7) -> str:
    """
    Calls Gemini API directly (no caching).
    Tries models in order of quota availability.
    On 429 quota errors, automatically falls back to the next model.
    Returns the generated text, or a user-friendly error string.
    """
    api_key = _get_api_key()
    if not api_key or api_key == "your_gemini_api_key_here":
        return "⚠️ **Gemini API key not configured.** Add your key to the `.env` file."

    # Explicitly unset GOOGLE_API_KEY for this call so the SDK uses our key
    old_google_key = os.environ.pop("GOOGLE_API_KEY", None)

    # Model priority list — ordered by quota availability
    MODELS_TO_TRY = [
        "gemini-2.5-flash",        # most capable, separate quota pool
        "gemini-2.0-flash-lite",   # lightest quota usage
        "gemini-2.0-flash",        # standard
    ]

    last_error = ""
    result = None
    try:
        from google import genai as genai_v2
        from google.genai import types as genai_types
        client = genai_v2.Client(api_key=api_key)

        for model_name in MODELS_TO_TRY:
            try:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=512,
                    ),
                )
                result = resp.text.strip()
                break
            except Exception as exc:
                last_error = str(exc)
                if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error or "quota" in last_error.lower():
                    logging.warning("Model %s quota exceeded, trying next…", model_name)
                    continue
                break   # non-quota error — stop
    finally:
        # Restore GOOGLE_API_KEY if it was set before
        if old_google_key is not None:
            os.environ["GOOGLE_API_KEY"] = old_google_key

    if result is not None:
        return result

    if "429" in last_error or "RESOURCE_EXHAUSTED" in last_error:
        return (
            "⚠️ **Gemini quota exceeded** on all available models. "
            "Free tier limits reset after a short wait. "
            "Click **🔄 Regenerate** in a minute, or "
            "[check your quota here](https://ai.google.dev/gemini-api/docs/rate-limits)."
        )
    if "404" in last_error or "not found" in last_error.lower():
        return (
            "⚠️ **Gemini model not available** for this API key. "
            "This is usually a key or region mismatch. "
            "Check your key in `.env` and try again."
        )
    return f"⚠️ Guidance generation failed: {last_error[:200]}"


@st.cache_resource(show_spinner="Checking Gemini connection…")
def load_gemini():
    """
    Returns True if a valid API key exists, False otherwise.
    Actual calls go through _call_gemini() which always reads the fresh key.
    """
    key = _get_api_key()
    return bool(key and key != "your_gemini_api_key_here")


# ── Session state ──────────────────────────────────────────────────────────
for k, v in {
    "result": None, "guidance": None,
    "role": "Learner", "subject": SUBJECT_OPTIONS[0],
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

predictor, load_error = load_predictor()
gemini_ready = load_gemini()   # bool: True if key present


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════

def _badge(label: str, cls: str = "badge-ok") -> str:
    return f'<span class="badge {cls}">{label}</span>'


def _generate_guidance(text: str, prediction: dict, role: str,
                       subject: str, temperature: float = 0.7) -> str:
    TONE = {
        "Confused":   "Be clarifying and patient. Break things down step by step.",
        "Frustrated": "Validate the struggle first, then offer a concise next step.",
        "Curious":    "Be enthusiastic. Encourage exploration; ask a follow-up question.",
        "Confident":  "Affirm their momentum and suggest a natural next challenge.",
        "Excited":    "Match their energy but channel it into one focused action.",
        "Anxious":    "Be calm and reassuring. Offer small, manageable next steps.",
        "Neutral":    "Be warm and factual. Provide clear, practical guidance.",
    }
    primary  = prediction["primary"]
    conf     = prediction["primary_confidence"]
    is_mixed = prediction["is_mixed"]
    secondary   = prediction.get("secondary")
    sec_conf = prediction.get("secondary_confidence")

    mixed_note = ""
    if is_mixed and secondary:
        mixed_note = (
            f"\nAlso showing signs of **{secondary}** ({sec_conf:.0%}). "
            "Acknowledge both emotional states."
        )

    role_line = (
        "You are responding directly to a **student** who needs support."
        if role == "Learner"
        else "You are advising an **educator/TA** on how best to respond."
    )

    prompt = f"""You are an empathetic AI learning coach.
{role_line}

Subject: {subject}
Detected emotion: **{primary}** ({conf:.0%} confidence){mixed_note}
Tone guidance: {TONE.get(primary, '')}

Student message:
\"\"\"{text}\"\"\"

Write a concise, empathetic, actionable response (3-5 sentences or short bullets).
- Open by acknowledging the emotional state.
- Give 1-2 concrete next steps tailored to {subject}.
- Keep language accessible and encouraging.
- Use light markdown (bold key terms, bullets where helpful).
"""

    if not gemini_ready:
        return (
            f"**Note:** Gemini API key not configured — add `GEMINI_API_KEY` to `.env`.\n\n"
            f"Detected emotion: **{primary}** ({conf:.0%} confidence)."
        )

    return _call_gemini(prompt, temperature=temperature)


def _log_interaction(result: dict, role: str, subject: str, guidance: str) -> None:
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        write_header = not LOG_FILE.exists()
        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if write_header:
                w.writeheader()
            bilstm = result.get("bilstm", {})
            bert   = result.get("bert",   {})
            final  = result.get("final",  {})
            w.writerow({
                "timestamp":       datetime.now().isoformat(timespec="seconds"),
                "input_text":      result.get("input_text", ""),
                "role":            role,
                "subject":         subject,
                "bilstm_primary":  bilstm.get("primary", ""),
                "bilstm_conf":     round(bilstm.get("primary_confidence", 0), 4),
                "bilstm_mixed":    bilstm.get("is_mixed", False),
                "bilstm_secondary":bilstm.get("secondary") or "",
                "bert_primary":    bert.get("primary", ""),
                "bert_conf":       round(bert.get("primary_confidence", 0), 4),
                "bert_mixed":      bert.get("is_mixed", False),
                "bert_secondary":  bert.get("secondary") or "",
                "final_primary":   final.get("primary", ""),
                "final_conf":      round(final.get("primary_confidence", 0), 4),
                "final_mixed":     final.get("is_mixed", False),
                "final_secondary": final.get("secondary") or "",
                "gemini_response": guidance.replace("\n", " "),
            })
    except Exception as exc:
        logging.error("Logging failed: %s", exc)


def _load_logs() -> pd.DataFrame:
    if not LOG_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(LOG_FILE)
        df["final_conf"] = pd.to_numeric(df.get("final_conf", pd.Series()), errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<p class="title">🧠 Emotion-Aware Learning Support Engine</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Paste a student message → detect emotions → '
    'get AI-generated personalised guidance.</p>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════════════════════════════════
# MODELS NOT READY — show setup instructions and stop
# ══════════════════════════════════════════════════════════════════════════
if predictor is None:
    st.error("⚠️ Models not ready — training required before the app can run.")
    st.markdown('<div class="setup-box">', unsafe_allow_html=True)
    st.markdown("### 🔧 Setup Instructions")
    st.markdown(
        "The trained model files were not found. "
        "Run the training script to generate them:\n"
    )
    st.code("# Train both models (recommended)\npython src/train.py --model both", language="bash")
    st.markdown("Or train each model separately:")
    st.code(
        "python src/train.py --model bilstm\n"
        "python src/train.py --model bert",
        language="bash",
    )
    st.markdown("**Expected output files after training:**")
    st.code(
        "models/bilstm/bilstm_emotion_model.h5\n"
        "models/bilstm/tokenizer.pkl\n"
        "models/bert_emotion_model_final/config.json\n"
        "models/bert_emotion_model_final/pytorch_model.bin",
        language="text",
    )
    if load_error:
        with st.expander("Technical error details"):
            st.code(load_error, language="text")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# ── Status badges (only shown when models are loaded) ──────────────────────
sc1, sc2, sc3, _ = st.columns([1.1, 1.3, 1.1, 5])
with sc1:
    st.markdown(_badge("BiLSTM ✓"), unsafe_allow_html=True)
with sc2:
    st.markdown(_badge("DistilBERT ✓"), unsafe_allow_html=True)
with sc3:
    lbl = "Gemini ✓" if gemini_ready else "Gemini — no key"
    st.markdown(
        f'<span class="badge" style="background:{"#1F8A5A" if gemini_ready else "#666"};color:#fff">'
        f'{lbl}</span>',
        unsafe_allow_html=True,
    )

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# INPUT PANEL
# ══════════════════════════════════════════════════════════════════════════
st.subheader("📝 Input")
inp_col, cfg_col = st.columns([3, 1])

with inp_col:
    user_text = st.text_area(
        "Student / educator message",
        placeholder=(
            "e.g. I keep re-reading this section on derivatives "
            "but I still don't understand why the chain rule works…"
        ),
        height=130,
        label_visibility="collapsed",
    )

with cfg_col:
    st.session_state.role    = st.selectbox("Role",    ["Learner", "Educator / TA"])
    st.session_state.subject = st.selectbox("Subject", SUBJECT_OPTIONS)

analyse_btn = st.button("🔍 Analyse", type="primary", width="stretch")

# ══════════════════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
if analyse_btn:
    if not user_text.strip():
        st.error("Please enter some text before analysing.")
    else:
        with st.spinner("Running emotion detection…"):
            result = predictor.predict(user_text)

        if not result["ok"]:
            st.error(result["error"])
        else:
            st.session_state.result = result
            with st.spinner("Generating AI guidance…"):
                guidance = _generate_guidance(
                    user_text,
                    result["final"],
                    st.session_state.role,
                    st.session_state.subject,
                )
            st.session_state.guidance = guidance
            _log_interaction(
                result,
                st.session_state.role,
                st.session_state.subject,
                guidance,
            )

# ══════════════════════════════════════════════════════════════════════════
# RESULTS PANEL
# ══════════════════════════════════════════════════════════════════════════
if st.session_state.result is not None:
    result = st.session_state.result
    st.divider()
    st.subheader("📊 Results")

    mc1, mc2 = st.columns(2)

    for col, key, title in [(mc1, "bilstm", "BiLSTM"), (mc2, "bert", "DistilBERT")]:
        r     = result[key]
        emoji = EMOTION_EMOJI.get(r["primary"], "")
        color = EMOTION_COLORS.get(r["primary"], "#888")

        with col:
            mixed_html = ""
            if r["is_mixed"]:
                mixed_html = (
                    f'&nbsp;<span class="badge badge-mixed">'
                    f'also {r["secondary"]} {r["secondary_confidence"]:.0%}</span>'
                )
            st.markdown(
                f'<div class="card">'
                f'<b>{title}</b><br>'
                f'<span style="font-size:1.6rem;color:{color}">{emoji} {r["primary"]}</span>'
                f'<br><span style="color:#AAAACC;font-size:.85rem">'
                f'{r["primary_confidence"]:.1%} confidence</span>'
                f'{mixed_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Horizontal bar chart — all 7 class scores
            scores_df = (
                pd.DataFrame({
                    "Emotion": list(r["all_scores"].keys()),
                    "Score":   list(r["all_scores"].values()),
                })
                .sort_values("Score", ascending=True)
            )
            fig = px.bar(
                scores_df, x="Score", y="Emotion", orientation="h",
                color="Emotion", color_discrete_map=EMOTION_COLORS,
                height=220,
            )
            fig.update_layout(
                margin=dict(l=0, r=0, t=4, b=4),
                showlegend=False,
                xaxis=dict(range=[0, 1], tickformat=".0%"),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#CCCCCC",
            )
            st.plotly_chart(fig, width="stretch")

    # ── Final prediction ────────────────────────────────────────────────────
    final   = result["final"]
    f_emoji = EMOTION_EMOJI.get(final["primary"], "")
    f_color = EMOTION_COLORS.get(final["primary"], "#888")
    mixed_html = ""
    if final["is_mixed"]:
        mixed_html = (
            f'&nbsp;<span class="badge badge-mixed">'
            f'mixed: {final["secondary"]} {final["secondary_confidence"]:.0%}</span>'
        )
    st.markdown(
        f'<div class="card" style="border:1px solid {f_color};">'
        f'<b>🏆 Final Prediction</b>&nbsp;&nbsp;'
        f'<span style="font-size:1.4rem;color:{f_color}">{f_emoji} {final["primary"]}</span>'
        f'&nbsp;<span style="color:#AAAACC;font-size:.9rem">'
        f'({final["primary_confidence"]:.1%})</span>'
        f'{mixed_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Guidance panel ──────────────────────────────────────────────────────
    st.divider()
    st.subheader("💬 AI Guidance")

    _, regen_col = st.columns([5, 1])
    with regen_col:
        regen = st.button("🔄 Regenerate", width="stretch")

    if regen:
        with st.spinner("Regenerating guidance…"):
            new_g = _generate_guidance(
                result["input_text"],
                result["final"],
                st.session_state.role,
                st.session_state.subject,
                temperature=0.9,
            )
        st.session_state.guidance = new_g
        _log_interaction(
            result, st.session_state.role, st.session_state.subject, new_g
        )

    st.markdown(
        f'<div class="guidance-box">{st.session_state.guidance}</div>',
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════
# ANALYTICS DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📈 Analytics")

df = _load_logs()
total = len(df)

m1, m2, m3, m4 = st.columns(4)
most_common = (
    df["final_primary"].mode().iloc[0]
    if total > 0 and "final_primary" in df.columns else "—"
)
mixed_rate = (
    df["final_mixed"].map({"True": True, "False": False}).fillna(False).mean()
    if total > 0 and "final_mixed" in df.columns else 0.0
)
avg_conf = (
    df["final_conf"].mean()
    if total > 0 and "final_conf" in df.columns else 0.0
)

for col, label, value in [
    (m1, "Total Analyses",     str(total)),
    (m2, "Top Emotion",        f"{EMOTION_EMOJI.get(most_common, '')} {most_common}"),
    (m3, "Mixed Emotion Rate", f"{mixed_rate:.0%}"),
    (m4, "Avg Confidence",     f"{avg_conf:.0%}"),
]:
    with col:
        st.metric(label, value)

ch1, ch2 = st.columns(2)

# Donut chart
with ch1:
    if total > 0 and "final_primary" in df.columns:
        counts = df["final_primary"].value_counts()
        for e in EMOTION_CLASSES:
            if e not in counts:
                counts[e] = 0
        fig_donut = go.Figure(go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.55,
            marker_colors=[EMOTION_COLORS.get(e, "#888") for e in counts.index],
            textinfo="percent+label",
        ))
        fig_donut.update_layout(
            title="Emotion Distribution", showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)", font_color="#CCCCCC",
            margin=dict(t=40, b=10, l=10, r=10),
        )
    else:
        fig_donut = go.Figure()
        fig_donut.add_annotation(
            text="No data yet — run some analyses first.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        )
        fig_donut.update_layout(
            title="Emotion Distribution",
            paper_bgcolor="rgba(0,0,0,0)", font_color="#CCCCCC",
        )
    st.plotly_chart(fig_donut, width="stretch")

# Confidence trend
with ch2:
    if total > 0 and "final_conf" in df.columns:
        plot_df = df.reset_index(drop=True)
        plot_df["#"] = plot_df.index + 1
        fig_trend = px.scatter(
            plot_df, x="#", y="final_conf",
            color="final_primary", color_discrete_map=EMOTION_COLORS,
            labels={
                "final_conf": "Confidence",
                "#": "Interaction #",
                "final_primary": "Emotion",
            },
            height=300,
        )
        rolling = plot_df["final_conf"].rolling(5, min_periods=1).mean()
        fig_trend.add_trace(go.Scatter(
            x=plot_df["#"], y=rolling, mode="lines", name="Avg(5)",
            line=dict(color="white", width=2, dash="dot"),
        ))
        fig_trend.update_layout(
            title="Confidence Over Time",
            yaxis=dict(range=[0, 1], tickformat=".0%"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#CCCCCC", margin=dict(t=40, b=20, l=40, r=10),
        )
    else:
        fig_trend = go.Figure()
        fig_trend.add_annotation(
            text="No data yet — run some analyses first.",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
        )
        fig_trend.update_layout(
            title="Confidence Over Time",
            paper_bgcolor="rgba(0,0,0,0)", font_color="#CCCCCC",
        )
    st.plotly_chart(fig_trend, width="stretch")
