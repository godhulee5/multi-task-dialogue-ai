
import streamlit as st
import pandas as pd
from utils.style import load_css
from utils.storage import get_records

st.set_page_config(page_title="Intent Dashboard", page_icon="🎯", layout="wide")
load_css()
st.title("🎯 Intent / Dialogue Act Dashboard")
records=get_records()
if not records:
    st.info("No saved analyses yet.")
else:
    df=pd.DataFrame(records)
    c1,c2,c3=st.columns(3)
    c1.metric("Total messages",len(df))
    c2.metric("Most common intent",df["dialogue_act"].mode().iat[0])
    c3.metric("High priority",int((df["priority"]=="High").sum()))
    a,b=st.columns(2)
    with a:
        st.subheader("Intent distribution")
        st.bar_chart(df["dialogue_act"].value_counts())
    with b:
        st.subheader("Intent vs emotion")
        st.bar_chart(pd.crosstab(df["dialogue_act"],df["emotion"]))
    st.subheader("Intent details")
    st.dataframe(df[["timestamp","text","dialogue_act","emotion","priority"]].tail(30),use_container_width=True,hide_index=True)
