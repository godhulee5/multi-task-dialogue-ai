
import streamlit as st
import pandas as pd
from utils.style import load_css
from utils.storage import get_records

st.set_page_config(page_title="Sentiment Dashboard", page_icon="😊", layout="wide")
load_css()
st.title("😊 Sentiment & Emotion Dashboard")
records=get_records()
if not records:
    st.info("No saved analyses yet. Use Live Support or Message Analyzer first.")
else:
    df=pd.DataFrame(records)
    c1,c2,c3=st.columns(3)
    c1.metric("Total analyzed",len(df))
    c2.metric("Most common",df["emotion"].mode().iat[0])
    c3.metric("Anger cases",int((df["emotion"]=="Anger").sum()))
    a,b=st.columns(2)
    with a:
        st.subheader("Emotion distribution")
        st.bar_chart(df["emotion"].value_counts())
    with b:
        st.subheader("Emotion × Priority")
        st.bar_chart(pd.crosstab(df["emotion"],df["priority"]))
    st.subheader("Recent analyzed messages")
    st.dataframe(df[["timestamp","text","emotion","dialogue_act","priority"]].tail(20),use_container_width=True,hide_index=True)
