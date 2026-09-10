
import streamlit as st
import pandas as pd
from utils.style import load_css
from utils.prediction import predict_text

st.set_page_config(page_title="Message Analyzer", page_icon="🔎", layout="wide")
load_css()
st.title("🔎 Customer Message Analyzer")
st.caption("Analyze one message and inspect both joint-task predictions.")

text = st.text_area("Customer message", height=150, placeholder="Example: I have been waiting for my order for two weeks!")
if st.button("🧠 Analyze Message", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Enter a customer message first.")
    else:
        r = predict_text(text)
        if r.get("demo"):
            st.warning("Model files are not installed yet. Showing demo mode. Copy your Colab model folder into models/multitask_model to enable real predictions.")
        a,b,c,d = st.columns(4)
        a.metric("Dialogue Act", r["dialogue_act"])
        b.metric("Emotion", r["emotion"])
        c.metric("Act Confidence", f'{r["dialogue_confidence"]:.1%}')
        d.metric("Priority", r["priority"])
        x,y = st.columns(2)
        with x:
            st.subheader("🎯 Dialogue Act probabilities")
            st.bar_chart(pd.DataFrame({"Probability":r["dialogue_probabilities"]}))
        with y:
            st.subheader("🎭 Emotion probabilities")
            st.bar_chart(pd.DataFrame({"Probability":r["emotion_probabilities"]}))
