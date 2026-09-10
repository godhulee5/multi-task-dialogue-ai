
import streamlit as st
import pandas as pd
from utils.style import load_css
from utils.storage import get_records

st.set_page_config(page_title="Priority & Escalation", page_icon="🚨", layout="wide")
load_css()
st.title("🚨 Priority & Escalation Center")
st.caption("Rule-based operational prioritization built on top of the joint-task predictions.")
records=get_records()
if not records:
    st.info("No analyzed messages yet.")
else:
    df=pd.DataFrame(records)
    high=df[df["priority"]=="High"]
    medium=df[df["priority"]=="Medium"]
    low=df[df["priority"]=="Low"]
    a,b,c=st.columns(3)
    a.metric("🔴 High",len(high))
    b.metric("🟡 Medium",len(medium))
    c.metric("🟢 Low",len(low))
    st.markdown("### 🔴 High Priority Cases")
    if high.empty: st.success("No high-priority cases.")
    else: st.dataframe(high[["timestamp","text","emotion","dialogue_act","priority"]],use_container_width=True,hide_index=True)
    st.markdown("### 🧭 Escalation logic")
    st.markdown("""
    - **High:** Anger or Disgust detected.
    - **Medium:** Fear, Sadness, or Directive.
    - **Low:** Other combinations.

    This is an application-level rule, not a trained escalation classifier.
    """)
