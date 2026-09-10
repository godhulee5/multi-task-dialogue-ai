
import streamlit as st
from pathlib import Path
from utils.style import load_css
from utils.storage import get_records

st.set_page_config(
    page_title="Support Intelligence AI",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)
load_css()

records = get_records()

with st.sidebar:
    st.markdown("## 🎧 Support Intelligence AI")
    st.caption("Customer Support Sentiment & Intent Tracker")
    st.markdown("---")
    st.markdown("### 🧠 AI Engine")
    st.info("DistilBERT Multi-Task Learning\n\nOne shared encoder predicts both Dialogue Act and Emotion.")
    st.markdown("---")
    st.caption("MSc Data Analytics • NLP • Multi-Task Learning • Streamlit")

st.markdown("""
<div class="hero">
    <div class="hero-badge">CUSTOMER SUPPORT INTELLIGENCE</div>
    <h1>Understand what customers say<br>and how they feel.</h1>
    <p>Analyze customer conversations using your trained DistilBERT joint-task model for real-time emotion and dialogue-act prediction.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### 📊 Support Overview")
c1,c2,c3,c4 = st.columns(4)
emotions = [r.get("emotion","Unknown") for r in records]
acts = [r.get("dialogue_act","Unknown") for r in records]
high = sum(r.get("priority") == "High" for r in records)

c1.metric("Messages analyzed", len(records))
c2.metric("Conversations", len(set(r.get("conversation_id","default") for r in records)) if records else 0)
c3.metric("High priority", high)
c4.metric("Most common emotion", max(set(emotions), key=emotions.count) if emotions else "—")

st.markdown("### 🚀 Application Modules")
cols = st.columns(4)
modules = [
    ("💬","Live Support","Chat with the AI and analyze every customer message."),
    ("🔎","Message Analyzer","Run a prediction on one customer message."),
    ("🧠","Conversation Analysis","Analyze a complete multi-message conversation."),
    ("📊","Analytics","Explore emotion, intent and escalation trends."),
]
for col,(icon,title,desc) in zip(cols,modules):
    with col:
        st.markdown(f'<div class="card"><div class="icon">{icon}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

st.markdown("### 🏗️ Multi-Task Architecture")
st.markdown("""
<div class="architecture">
  <div class="archbox">💬<br><b>Customer Message</b></div>
  <div class="arrow">→</div>
  <div class="archbox">🤖<br><b>DistilBERT<br>Shared Encoder</b></div>
  <div class="arrow">→</div>
  <div class="archbox">🔀<br><b>Two Task Heads</b></div>
  <div class="arrow">→</div>
  <div class="archbox">😊 Emotion<br>🎯 Dialogue Act</div>
</div>
""", unsafe_allow_html=True)

st.markdown("### ✨ What this system provides")
a,b,c = st.columns(3)
with a:
    st.markdown('<div class="mini-card"><b>🎭 Emotion Intelligence</b><br><span>Neutral, Anger, Disgust, Fear, Happiness, Sadness and Surprise.</span></div>', unsafe_allow_html=True)
with b:
    st.markdown('<div class="mini-card"><b>🎯 Intent / Dialogue Act</b><br><span>Inform, Question, Directive and Commissive predictions.</span></div>', unsafe_allow_html=True)
with c:
    st.markdown('<div class="mini-card"><b>🚨 Support Prioritization</b><br><span>Identify conversations that may need human attention.</span></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Support Intelligence AI • Powered by your DistilBERT Multi-Task Model</div>', unsafe_allow_html=True)
