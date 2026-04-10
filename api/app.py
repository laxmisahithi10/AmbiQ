import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

import streamlit as st

from rule_engine.rule_engine import analyze_question, get_vague_words
from ml_model.ml_predictor import AmbiguityMLPredictor

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Question Ambiguity Analysis System",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
    /* Background */
    .stApp { background-color: #0f1117; color: #e0e0e0; }

    /* Title */
    h1 { text-align: center; color: #7c83fd; font-size: 2.2rem; padding-bottom: 0.2rem; }

    /* Subtitle */
    .subtitle {
        text-align: center; color: #9a9a9a; font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    /* Section Cards */
    .card {
        background-color: #1e2130;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.2rem;
        border: 1px solid #2e3250;
    }
    .card h3 { color: #7c83fd; margin-top: 0; font-size: 1.05rem; }

    /* Metric rows */
    .metric-row { display: flex; gap: 1rem; margin-top: 0.5rem; flex-wrap: wrap; }
    .metric-box {
        background: #262b3e;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        flex: 1;
        min-width: 120px;
        text-align: center;
    }
    .metric-box .label { font-size: 0.72rem; color: #9a9a9a; text-transform: uppercase; }
    .metric-box .value { font-size: 1.1rem; font-weight: 700; color: #e0e0e0; }

    /* Intent badge */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 0.5rem;
    }
    .badge-booking   { background: #1a3a5c; color: #5bc0eb; }
    .badge-purchase  { background: #2a1a5c; color: #a78bfa; }
    .badge-task      { background: #1a3a2a; color: #4ade80; }
    .badge-info      { background: #2a2a1a; color: #fbbf24; }
    .badge-unknown   { background: #2a2a2a; color: #9a9a9a; }

    /* Verdict boxes */
    .verdict-clear {
        background: #0d2b1a; border: 1px solid #4ade80;
        border-radius: 12px; padding: 1rem 1.5rem; text-align: center;
        color: #4ade80; font-size: 1.2rem; font-weight: 700;
    }
    .verdict-ambiguous {
        background: #2b0d0d; border: 1px solid #f87171;
        border-radius: 12px; padding: 1rem 1.5rem; text-align: center;
        color: #f87171; font-size: 1.2rem; font-weight: 700;
    }

    /* Reason bullets */
    .reason-item {
        background: #262b3e; border-left: 3px solid #7c83fd;
        border-radius: 6px; padding: 0.4rem 0.8rem;
        margin: 0.3rem 0; font-size: 0.88rem; color: #c0c0c0;
    }

    /* Confidence bar label */
    .conf-label { font-size: 0.82rem; color: #9a9a9a; margin-bottom: 0.2rem; }

    /* Input label override */
    label { color: #c0c0c0 !important; }

    /* Highlight */
    .highlight { background: #7c2d2d; color: #fca5a5; border-radius: 4px; padding: 0 4px; font-weight: 600; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #13151f; }
    .sidebar-title { color: #7c83fd; font-size: 1rem; font-weight: 700; margin-bottom: 0.5rem; }
    .sidebar-label { color: #9a9a9a; font-size: 0.75rem; text-transform: uppercase; margin: 0.8rem 0 0.3rem 0; }

    /* Footer */
    .footer { text-align: center; color: #555; font-size: 0.78rem; margin-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Load ML model
# --------------------------------------------------
@st.cache_resource
def load_ml_model():
    return AmbiguityMLPredictor(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_model/ambiguity_model_quora.pkl"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_model/tfidf_vectorizer_quora.pkl")
    )

ml_predictor = load_ml_model()

# --------------------------------------------------
# Sidebar — Example Questions
# --------------------------------------------------
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

with st.sidebar:
    st.markdown("<div class='sidebar-title'>💡 Try Example Questions</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-label'>✅ Clear</div>", unsafe_allow_html=True)
    clear_examples = [
        "What is machine learning?",
        "How do I install Python?",
        "Who invented the telephone?",
        "What are the benefits of exercise?",
        "How does photosynthesis work?",
    ]
    for ex in clear_examples:
        if st.button(ex, key=f"c_{ex}"):
            st.session_state.selected_question = ex

    st.markdown("<div class='sidebar-label'>⚠️ Ambiguous</div>", unsafe_allow_html=True)
    ambiguous_examples = [
        "Book it",
        "Send that file",
        "Delete this",
        "Order now",
        "Schedule a meeting",
        "Fix the bug",
    ]
    for ex in ambiguous_examples:
        if st.button(ex, key=f"a_{ex}"):
            st.session_state.selected_question = ex

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown("<h1>🧠 Question Ambiguity Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='subtitle'>Detects whether a question is <b>Clear</b> or <b>Ambiguous</b> "
    "using Rule-Based NLP + Machine Learning</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Input
# --------------------------------------------------
question = st.text_input(
    "Enter your question",
    value=st.session_state.selected_question,
    placeholder="Type your question here... e.g. Book it / What is AI?",
    label_visibility="collapsed"
)

# --------------------------------------------------
# Analysis
# --------------------------------------------------
if question and question.strip():

    rule_result = analyze_question(question)
    ml_result   = ml_predictor.predict(question)

    # ── Ambiguity Highlighting ────────────────────
    vague_words = get_vague_words(question)
    if vague_words:
        highlighted = question
        for w in set(vague_words):
            highlighted = highlighted.replace(w, f"<span class='highlight'>{w}</span>")
        st.markdown(
            f"<div class='card'><h3>🖊️ Ambiguity Highlighting</h3>"
            f"<p style='font-size:1.05rem'>{highlighted}</p>"
            f"<p style='color:#9a9a9a;font-size:0.8rem'>Highlighted words are vague references detected by WordNet-expanded analysis.</p></div>",
            unsafe_allow_html=True
        )

    badge_class = {
        "booking": "badge-booking",
        "purchase": "badge-purchase",
        "task": "badge-task",
        "informational": "badge-info",
    }.get(rule_result.intent, "badge-unknown")

    # ── Rule Engine Card ──────────────────────────
    st.markdown(f"""
    <div class='card'>
        <h3>🔍 Rule-Based NLP Analysis</h3>
        <div class='metric-row'>
            <div class='metric-box'>
                <div class='label'>Decision</div>
                <div class='value'>{rule_result.label}</div>
            </div>
            <div class='metric-box'>
                <div class='label'>Intent</div>
                <div class='value'>{rule_result.intent.capitalize()}</div>
            </div>
            <div class='metric-box'>
                <div class='label'>Score</div>
                <div class='value'>{rule_result.score}</div>
            </div>
            <div class='metric-box'>
                <div class='label'>Threshold</div>
                <div class='value'>{rule_result.threshold}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if rule_result.triggered_rules:
        reasons_html = "".join(
            f"<div class='reason-item'>• {r.reason}</div>"
            for r in rule_result.triggered_rules
        )
        st.markdown(f"<div class='card'><h3>📋 Triggered Rules</h3>{reasons_html}</div>", unsafe_allow_html=True)

    # ── ML Card ───────────────────────────────────
    st.markdown(f"""
    <div class='card'>
        <h3>🤖 ML-Based Analysis</h3>
        <div class='metric-row'>
            <div class='metric-box'>
                <div class='label'>Prediction</div>
                <div class='value'>{ml_result['label']}</div>
            </div>
            <div class='metric-box'>
                <div class='label'>Confidence</div>
                <div class='value'>{ml_result['confidence']:.1f}%</div>
            </div>
            <div class='metric-box'>
                <div class='label'>Clear %</div>
                <div class='value'>{ml_result['probabilities']['clear']:.1f}%</div>
            </div>
            <div class='metric-box'>
                <div class='label'>Ambiguous %</div>
                <div class='value'>{ml_result['probabilities']['ambiguous']:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='conf-label'>ML Ambiguity Confidence</div>", unsafe_allow_html=True)
    st.progress(int(ml_result['probabilities']['ambiguous']))

    # ── Final Verdict ─────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    is_ambiguous = (
        rule_result.label == "Ambiguous"
        if rule_result.intent in ["booking", "purchase", "task"]
        else rule_result.label == "Ambiguous" or ml_result["is_ambiguous"]
    )

    if is_ambiguous:
        st.markdown("<div class='verdict-ambiguous'>⚠️ The question is AMBIGUOUS</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        triggered_names = {r.name for r in rule_result.triggered_rules}
        clarification_inputs = {}

        st.markdown("<div class='card'><h3>💬 Please fill in the missing details</h3></div>", unsafe_allow_html=True)

        if "MissingRequiredObject" in triggered_names or "ActionWithoutObject" in triggered_names or "TaskWithVagueObject" in triggered_names:
            if rule_result.intent == "booking":
                clarification_inputs["object"] = st.text_input("📌 What do you want to book?", placeholder="e.g. hotel, flight, table")
            elif rule_result.intent == "purchase":
                clarification_inputs["object"] = st.text_input("📌 What do you want to order/buy?", placeholder="e.g. pizza, laptop")
            else:
                clarification_inputs["object"] = st.text_input("📌 What is the target of this action?", placeholder="e.g. the report, the file")

        if "MissingRequiredDateTime" in triggered_names or "MissingTime" in triggered_names:
            clarification_inputs["datetime"] = st.text_input("📅 When?", placeholder="e.g. tomorrow at 3pm, next Monday")

        if "MissingRequiredLocation" in triggered_names or "MissingLocation" in triggered_names:
            clarification_inputs["location"] = st.text_input("📍 Where?", placeholder="e.g. Hyderabad, Room 101, online")

        if "MissingRequiredQuantity" in triggered_names:
            clarification_inputs["quantity"] = st.text_input("🔢 How many?", placeholder="e.g. 2, a dozen")

        if "VaguePronoun" in triggered_names or "TaskWithVagueObject" in triggered_names:
            clarification_inputs["vague"] = st.text_input("❓ What does 'it / this / that' refer to?", placeholder="e.g. the invoice, my laptop")

        filled = {k: v for k, v in clarification_inputs.items() if v and v.strip()}

        if filled and st.button("🔄 Re-analyze with clarifications"):
            refined_question = f"{question} {' '.join(filled.values())}"

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"<div class='card'><h3>🔁 Re-Analysis</h3><p style='color:#9a9a9a'>Refined: <b style='color:#e0e0e0'>{refined_question}</b></p></div>", unsafe_allow_html=True)

            refined_rule = analyze_question(refined_question)
            refined_ml   = ml_predictor.predict(refined_question)

            st.markdown(f"""
            <div class='card'>
                <div class='metric-row'>
                    <div class='metric-box'>
                        <div class='label'>Rule Engine</div>
                        <div class='value'>{refined_rule.label}</div>
                    </div>
                    <div class='metric-box'>
                        <div class='label'>ML Model</div>
                        <div class='value'>{refined_ml['label']}</div>
                    </div>
                    <div class='metric-box'>
                        <div class='label'>Confidence</div>
                        <div class='value'>{refined_ml['confidence']:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            original_triggered = {r.name for r in rule_result.triggered_rules}
            refined_triggered   = {r.name for r in refined_rule.triggered_rules}
            resolved_all_params = not (original_triggered & refined_triggered)

            refined_ambiguous = (
                not resolved_all_params and refined_rule.label == "Ambiguous"
                if refined_rule.intent in ["booking", "purchase", "task"]
                else refined_rule.label == "Ambiguous" or refined_ml["is_ambiguous"]
            )

            if refined_ambiguous:
                st.markdown("<div class='verdict-ambiguous'>⚠️ Still AMBIGUOUS — please provide more details</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='verdict-clear'>✅ Question is now CLEAR after clarification!</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='verdict-clear'>✅ The question is CLEAR</div>", unsafe_allow_html=True)

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("<div class='footer'>Mini Project &nbsp;•&nbsp; NLP + ML &nbsp;•&nbsp; Question Ambiguity Detection</div>", unsafe_allow_html=True)
