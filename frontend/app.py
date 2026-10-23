from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", st.secrets.get("API_URL", "http://localhost:8000"))

st.set_page_config(page_title="Toxic Comment Classifier", page_icon="🚦", layout="centered")
st.title("Toxic Comment Classifier")
st.caption("AI moderation dashboard for toxic comment detection")

comment = st.text_area("Comment to analyze", height=180, placeholder="Enter a comment to inspect its toxicity profile...")

if st.button("Analyze comment", type="primary"):
    if not comment.strip():
        st.warning("Please enter a comment before running analysis.")
    else:
        try:
            response = requests.post(f"{API_URL}/predict", json={"comment": comment}, timeout=30)
        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")
        else:
            if response.status_code == 200:
                data = response.json()
                st.subheader("Analysis result")
                col1, col2 = st.columns(2)
                col1.metric("Overall toxicity", f"{data['toxicity_score'] * 100:.1f}%")
                col2.metric("Moderation action", data["moderation_action"])

                st.write("Category probabilities:")
                for key, value in data["categories"].items():
                    st.progress(min(value, 1.0), text=f"{key}: {value * 100:.1f}%")

                st.info(f"Decision: {'Toxic' if data['is_toxic'] else 'Safe'}")
            else:
                st.error(f"Prediction failed with status {response.status_code}: {response.text}")

