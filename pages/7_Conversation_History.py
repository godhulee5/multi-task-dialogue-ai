
import streamlit as st
import pandas as pd
from utils.style import load_css
from utils.storage import get_records, clear_records

st.set_page_config(page_title="Conversation History", page_icon="📜", layout="wide")
load_css()
st.title("📜 Conversation History")
records=get_records()
if st.button("🗑️ Clear all saved history"):
    clear_records()
    st.success("History cleared.")
    st.rerun()
if not records:
    st.info("No saved messages yet.")
else:
    df=pd.DataFrame(records)
    q=st.text_input("🔎 Search messages")
    if q:
        df=df[df["text"].str.contains(q,case=False,na=False)]
    e=st.multiselect("Filter emotion",sorted(df["emotion"].unique()))
    if e: df=df[df["emotion"].isin(e)]
    st.dataframe(df.sort_values("timestamp",ascending=False),use_container_width=True,hide_index=True)
