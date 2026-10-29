from __future__ import annotations

import math
from typing import Any

import streamlit as st

from src.config import LABELS
from src.metrics import calculate_toxicity_score, determine_moderation_action
from src.predict import load_model, load_tokenizer, predict_comment


@st.cache_resource(show_spinner=False)
def get_model_runtime() -> tuple[Any, Any]:
    model = load_model()
    tokenizer = load_tokenizer()
    model.to("cpu")
    model.eval()
    return model, tokenizer


@st.cache_data(show_spinner=False)
def run_local_inference(comment: str) -> dict[str, Any]:
    categories = predict_comment(comment)
    score = calculate_toxicity_score(categories)
    action = determine_moderation_action(score)
    return {
        "is_toxic": score >= 0.5,
        "toxicity_score": score,
        "moderation_action": action,
        "model_version": "distilbert_tuned_experimental",
        "categories": categories,
    }


def normalize_categories(categories: dict[str, Any]) -> dict[str, float]:
    if not isinstance(categories, dict):
        raise ValueError("Prediction categories must be a dictionary.")

    ordered: dict[str, float] = {}
    for label in LABELS:
        if label not in categories:
            raise ValueError(f"Expected label '{label}' in prediction payload.")
        value = categories[label]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Category '{label}' must be numeric.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Category '{label}' must be finite.")
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"Category '{label}' must be in [0, 1].")
        ordered[label] = numeric
    return ordered


def render_category_probabilities(categories: dict[str, float]) -> None:
    st.subheader("Six-label toxicity probabilities")
    for label in LABELS:
        value = float(categories[label])
        label_text = label.replace("_", " ").title()
        status = "positive" if value >= 0.5 else "below threshold"
        st.markdown(f"**{label_text}**")
        st.progress(min(max(value, 0.0), 1.0), text=f"{value * 100:.1f}%  •  {status}")


def render_model_notice() -> None:
    st.caption("Model: tuned experimental DistilBERT checkpoint loaded directly from models/distilbert_experiments/exp_v1_tuned/")
    st.info("This checkpoint is experimental and should be treated as a local integration model rather than a production-ready classifier.")


def main() -> None:
    st.set_page_config(page_title="Toxic Comment Classification System", page_icon="🚦", layout="wide")
    st.title("Toxic Comment Classification System")
    st.caption("AI-powered toxicity detection for moderation and review workflows.")
    render_model_notice()

    with st.sidebar:
        st.markdown("### Runtime")
        st.caption("Inference runs locally in the Streamlit app with the tuned DistilBERT checkpoint.")
        st.caption("No separate FastAPI server is required on Streamlit Community Cloud.")

    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.markdown("### Analyze a comment")
        user_comment = st.text_area(
            "Comment",
            value=st.session_state.get("comment", ""),
            height=220,
            max_chars=5000,
            placeholder="Enter a comment to analyze for toxic content.",
        )
        st.session_state["comment"] = user_comment

        char_count = len(user_comment)
        st.caption(f"{char_count}/5000 characters")

        analyze_disabled = not user_comment.strip() or len(user_comment.strip()) > 5000
        analyze_clicked = st.button("Analyze Comment", type="primary", disabled=analyze_disabled)
        reset_clicked = st.button("Reset")

        if reset_clicked:
            for key in ["comment", "result"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        if analyze_clicked:
            if not user_comment.strip():
                st.warning("Please enter a comment before running analysis.")
            elif len(user_comment.strip()) > 5000:
                st.warning("The comment exceeds 5000 characters.")
            else:
                with st.spinner("Loading model and running inference..."):
                    try:
                        result = run_local_inference(user_comment)
                        st.session_state["result"] = result
                    except (TypeError, ValueError, FileNotFoundError) as exc:
                        st.error(f"Prediction failed: {exc}")
                        st.session_state["result"] = {"error": True, "message": str(exc)}
                    except Exception as exc:
                        st.error(f"Unexpected prediction error: {exc}")
                        st.session_state["result"] = {"error": True, "message": str(exc)}

    with right_col:
        st.markdown("### Overview")
        st.write("This interface runs the existing DistilBERT model directly in Streamlit and displays the six toxicity probabilities returned by the checkpoint.")
        with st.expander("Six toxicity categories"):
            categories = {
                "toxic": "General toxicity or aggressive, harmful language.",
                "severe_toxic": "Severe or highly abusive content.",
                "obscene": "Profanity or explicit offensive language.",
                "threat": "Threatening or violent language.",
                "insult": "Insulting or demeaning content.",
                "identity_hate": "Hate speech targeting a protected identity.",
            }
            for key, description in categories.items():
                st.markdown(f"- **{key.replace('_', ' ').title()}**: {description}")

    result = st.session_state.get("result")
    if result and result.get("error"):
        st.warning(result.get("message", "The prediction request failed."))
    elif result:
        st.markdown("---")
        st.subheader("Prediction summary")
        overall_status = "Toxic" if result.get("is_toxic") else "Non-toxic"
        score = float(result.get("toxicity_score", 0.0))
        status_color = "#d94f70" if result.get("is_toxic") else "#2ecc71"
        st.markdown(
            f"<div style='padding: 1rem; border-left: 6px solid {status_color}; background: rgba(255,255,255,0.02); border-radius: 8px;'>"
            f"<strong>Status:</strong> {overall_status}<br>"
            f"<strong>Overall score:</strong> {score:.4f}<br>"
            f"<strong>Moderation action:</strong> {result.get('moderation_action', 'unknown')}"
            f"</div>",
            unsafe_allow_html=True,
        )
        render_category_probabilities(result["categories"])

        with st.expander("Detailed prediction payload"):
            st.json({k: v for k, v in result.items() if k != "categories"})
            st.json(result["categories"])


if __name__ == "__main__":
    main()

