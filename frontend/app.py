from __future__ import annotations

import math
import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
DEFAULT_API_URL = API_URL
LABELS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def get_api_url() -> str:
    env_url = os.getenv("API_URL")
    if env_url:
        return env_url
    try:
        return st.secrets.get("API_URL", DEFAULT_API_URL)
    except Exception:
        return DEFAULT_API_URL


def get_label_order(categories: dict[str, Any]) -> list[str]:
    if not isinstance(categories, dict):
        raise ValueError("Prediction categories must be a dictionary.")
    ordered = []
    for label in LABELS:
        if label in categories:
            ordered.append(label)
    if len(ordered) != len(LABELS):
        raise ValueError(f"Expected six category labels, received {len(ordered)}.")
    return ordered


def normalize_categories(categories: dict[str, Any]) -> dict[str, float]:
    ordered_labels = get_label_order(categories)
    normalized: dict[str, float] = {}
    for label in ordered_labels:
        value = categories.get(label)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Category '{label}' must be a numeric probability.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"Category '{label}' must be finite.")
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"Category '{label}' probability must be in the range [0, 1].")
        normalized[label] = numeric
    return normalized


def validate_prediction_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Prediction response must be a JSON object.")

    if "categories" not in payload:
        raise ValueError("Prediction response is missing the 'categories' field.")
    categories = normalize_categories(payload["categories"])

    is_toxic = payload.get("is_toxic")
    if not isinstance(is_toxic, bool):
        raise ValueError("Prediction response is missing a valid boolean 'is_toxic' value.")

    toxicity_score = payload.get("toxicity_score")
    if not isinstance(toxicity_score, (int, float)) or isinstance(toxicity_score, bool):
        raise ValueError("Prediction response is missing a valid numeric 'toxicity_score'.")
    toxicity_score = float(toxicity_score)
    if not math.isfinite(toxicity_score):
        raise ValueError("toxicity_score must be finite.")

    moderation_action = payload.get("moderation_action")
    if not isinstance(moderation_action, str):
        raise ValueError("Prediction response is missing a valid 'moderation_action'.")

    model_version = payload.get("model_version", "unknown")
    if not isinstance(model_version, str):
        model_version = str(model_version)

    return {
        "is_toxic": is_toxic,
        "toxicity_score": toxicity_score,
        "moderation_action": moderation_action,
        "model_version": model_version,
        "categories": categories,
    }


def render_category_probabilities(categories: dict[str, float]) -> None:
    st.subheader("Six-label toxicity probabilities")
    for label in LABELS:
        value = float(categories[label])
        label_text = label.replace("_", " ").title()
        status = "positive" if value >= 0.5 else "below threshold"
        st.markdown(f"**{label_text}**")
        st.progress(min(max(value, 0.0), 1.0), text=f"{value * 100:.1f}%  •  {status}")


def render_model_status(model_version: str | None) -> None:
    label = model_version or "unknown"
    st.caption(f"Current backend model: {label}")
    if label in {"distilbert_debug_experimental", "distilbert_tuned_experimental"}:
        st.info("This checkpoint is an experimental local model for integration testing. Predictions are probabilistic and may be imperfect.")


def clear_results() -> None:
    for key in ["comment", "result"]:
        if key in st.session_state:
            del st.session_state[key]


def main() -> None:
    st.set_page_config(page_title="Toxic Comment Classification System", page_icon="🚦", layout="wide")
    st.title("Toxic Comment Classification System")
    st.caption("AI-powered toxicity detection for multilingual moderation and review workflows.")

    left_col, right_col = st.columns([2, 1])
    with left_col:
        st.markdown("### Analyze a comment")
        prompt = "Enter a comment to analyze for toxic content."
        user_comment = st.text_area(
            "Comment",
            value=st.session_state.get("comment", ""),
            height=220,
            max_chars=5000,
            placeholder=prompt,
        )
        st.session_state["comment"] = user_comment

        char_count = len(user_comment)
        st.caption(f"{char_count}/5000 characters")
        if char_count > 5000:
            st.warning("Comment exceeds the backend input limit.")

        button_col, reset_col = st.columns([1, 1])
        with button_col:
            analyze_disabled = not user_comment.strip() or len(user_comment.strip()) > 5000
            analyze_clicked = st.button("Analyze Comment", type="primary", disabled=analyze_disabled)
        with reset_col:
            reset_clicked = st.button("Reset")

        if reset_clicked:
            clear_results()
            st.session_state["comment"] = ""
            st.rerun()

        if analyze_clicked:
            if not user_comment.strip():
                st.warning("Please enter a comment before running analysis.")
            elif len(user_comment.strip()) > 5000:
                st.warning("The backend accepts comments up to 5000 characters. Please shorten the input.")
            else:
                with st.spinner("Running inference..."):
                    api_url = get_api_url()
                    try:
                        response = requests.post(
                            f"{api_url}/predict",
                            json={"comment": user_comment},
                            timeout=20,
                        )
                    except requests.RequestException as exc:
                        st.error(f"Unable to reach the API at {api_url}. Please confirm the backend is running. Error: {exc}")
                        st.session_state["result"] = {"error": True, "message": str(exc)}
                    else:
                        if response.status_code != 200:
                            message = response.text
                            try:
                                payload = response.json()
                                message = payload.get("detail", message)
                            except Exception:
                                pass
                            st.error(f"Prediction failed (HTTP {response.status_code}): {message}")
                            st.session_state["result"] = {"error": True, "message": str(message)}
                        else:
                            try:
                                payload = response.json()
                                result = validate_prediction_payload(payload)
                            except ValueError as exc:
                                st.error(f"Received an invalid prediction payload: {exc}")
                                st.session_state["result"] = {"error": True, "message": str(exc)}
                            except Exception as exc:
                                st.error(f"Could not parse the API response: {exc}")
                                st.session_state["result"] = {"error": True, "message": str(exc)}
                            else:
                                st.session_state["result"] = result

    with right_col:
        st.markdown("### Overview")
        st.write("This interface sends a comment to the backend and displays the six toxicity probabilities returned by the model.")
        st.write("The current model is experimental and should be treated as a local integration checkpoint, not a production-ready classifier.")
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
        st.caption("This value follows the backend's existing semantics. It is not a new calibrated probability metric invented by the frontend.")
        render_model_status(result.get("model_version"))
        render_category_probabilities(result["categories"])

        with st.expander("Detailed prediction payload"):
            st.json({k: v for k, v in result.items() if k != "categories"})
            st.json(result["categories"])


if __name__ == "__main__":
    main()

