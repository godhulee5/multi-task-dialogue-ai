
import streamlit as st
from datetime import datetime
from utils.style import load_css
from utils.prediction import predict_text, support_reply
from utils.storage import save_record

st.set_page_config(page_title="Live Support", page_icon="💬", layout="wide")
load_css()
st.title("💬 Live Customer Support")
st.caption("Real-time customer message analysis using your joint DistilBERT model.")

if "live_messages" not in st.session_state:
    st.session_state.live_messages = []

for m in st.session_state.live_messages:
    with st.chat_message(m["role"]):
        st.write(m["text"])
        if m["role"] == "assistant" and "result" in m:
            r = m["result"]
            st.caption(f'🎯 {r["dialogue_act"]}  •  🎭 {r["emotion"]}  •  🚨 {r["priority"]}')

prompt = st.chat_input("Type a customer message...")
if prompt:
    try:
        result = predict_text(prompt)
        st.session_state.live_messages.append({"role":"user","text":prompt})
        st.session_state.live_messages.append({"role":"assistant","text":support_reply(result),"result":result})
        save_record({
            "conversation_id": "live_" + datetime.now().strftime("%Y%m%d%H%M%S"),
            "text": prompt, "emotion": result["emotion"], "dialogue_act": result["dialogue_act"],
            "priority": result["priority"], "emotion_confidence": result["emotion_confidence"],
            "dialogue_confidence": result["dialogue_confidence"]
        })
        st.rerun()
    except Exception as e:
        st.error(str(e))

if st.button("🗑️ Clear current chat"):
    st.session_state.live_messages = []
    st.rerun()
