
import streamlit as st
import pandas as pd
from utils.style import load_css
from utils.prediction import predict_text

st.set_page_config(page_title="Conversation Analysis", page_icon="🧠", layout="wide")
load_css()
st.title("🧠 Complete Conversation Analysis")
st.caption("Paste a conversation with one customer message per line.")

conversation = st.text_area("Conversation", height=300, placeholder="Customer: Where is my order?\nCustomer: It was supposed to arrive yesterday!\nCustomer: Can you refund the delivery charge?")
if st.button("Analyze Conversation", type="primary"):
    lines = [x.strip() for x in conversation.splitlines() if x.strip()]
    if not lines:
        st.warning("Enter at least one message.")
    else:
        rows=[]
        for i,line in enumerate(lines,1):
            text=line.split(":",1)[1].strip() if ":" in line else line
            r=predict_text(text)
            rows.append({"Message #":i,"Message":text,"Dialogue Act":r["dialogue_act"],"Emotion":r["emotion"],"Priority":r["priority"],
                         "Act Confidence":r["dialogue_confidence"],"Emotion Confidence":r["emotion_confidence"]})
        df=pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        c1,c2,c3,c4=st.columns(4)
        c1.metric("Messages",len(df))
        c2.metric("Dominant emotion",df["Emotion"].mode().iat[0])
        c3.metric("Main intent",df["Dialogue Act"].mode().iat[0])
        c4.metric("High priority",int((df["Priority"]=="High").sum()))
        st.subheader("📈 Emotion progression")
        st.line_chart(pd.crosstab(df.index,df["Emotion"]).astype(int))
        st.subheader("🎯 Intent distribution")
        st.bar_chart(df["Dialogue Act"].value_counts())
