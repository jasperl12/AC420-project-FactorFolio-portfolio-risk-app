# FactorFolio - Data Check page (built in Step 2)
# Will show: raw returns, describe() summary stats, row counts, outlier boxplot, checklist
import streamlit as st

st.set_page_config(page_title="Data Check | FactorFolio", layout="centered")
st.title("Data Check")
st.write("Look at the data first: raw returns, summary statistics, outliers and missing values.")
st.info("Coming in Step 2 of the build plan.")
