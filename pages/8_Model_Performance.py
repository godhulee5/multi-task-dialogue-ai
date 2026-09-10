
import streamlit as st
import pandas as pd
from pathlib import Path
import json
from utils.style import load_css
from utils.model_utils import load_metadata, model_ready, MODEL_DIR

st.set_page_config(page_title="Model Performance", page_icon="📈", layout="wide")
load_css()
st.title("📈 Model Performance & Research")
st.caption("Metrics saved from your Google Colab training notebook.")

dialogue, emotion, metrics, config = load_metadata()
if not metrics:
    st.warning("metrics.json was not found. Copy the exported model folder from Colab into models/multitask_model.")
else:
    d=metrics.get("dialogue_act",{})
    e=metrics.get("emotion",{})
    st.markdown("### 🎯 Dialogue Act Classification")
    a,b,c,dcol=st.columns(4)
    a.metric("Accuracy",f'{d.get("accuracy",0):.2%}')
    b.metric("Precision",f'{d.get("precision",0):.2%}')
    c.metric("Recall",f'{d.get("recall",0):.2%}')
    dcol.metric("F1 Score",f'{d.get("f1",0):.2%}')
    st.markdown("### 🎭 Emotion Classification")
    a,b,c,dcol=st.columns(4)
    a.metric("Accuracy",f'{e.get("accuracy",0):.2%}')
    b.metric("Precision",f'{e.get("precision",0):.2%}')
    c.metric("Recall",f'{e.get("recall",0):.2%}')
    dcol.metric("F1 Score",f'{e.get("f1",0):.2%}')
    st.markdown("### ⚙️ Training Configuration")
    st.json(metrics.get("training",{}))
    st.markdown("### 🏗️ Model Architecture")
    st.code("""Input Utterance
      ↓
Tokenizer (max_length=64)
      ↓
DistilBERT Shared Encoder
      ↓
[CLS] representation + Dropout(0.3)
      ├── Dialogue Act Classifier → 4 classes
      └── Emotion Classifier      → 7 classes
""")
    st.success("The application uses the same joint-task architecture as the supplied Colab notebook.")

st.markdown("### 📦 Model files")
expected=["model.pt","labels.json","config.json","metrics.json","tokenizer_config.json"]
for name in expected:
    st.write(("✅" if (MODEL_DIR/name).exists() else "⬜"), name)
