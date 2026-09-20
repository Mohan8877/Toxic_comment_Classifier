# Toxic Comment Classifier

End-to-end skeleton for a toxic comment classification system using deep learning (Bi-LSTM) with a FastAPI backend and simple Streamlit frontend.

Quick start

1. Create virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run API (development):

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

3. Run frontend demo:

```bash
streamlit run frontend/app.py
```

Project layout is scaffolded to implement training, prediction API, DB storage, and an admin dashboard.
# Toxic_comment_Classifier
