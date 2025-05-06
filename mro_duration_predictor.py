
import streamlit as st
import numpy as np

st.set_page_config(page_title="MRO Duration Predictor", layout="centered")

st.title("🛠️ MRO Duration Forecast Calculator")
st.markdown("""
This tool predicts the expected **MRO turnaround time (in days)** based on aircraft age and MRO region,
using your validated OLS regression model (R² = 0.74).
""")

# --- User inputs
age = st.slider("Aircraft Age (years)", min_value=1, max_value=30, value=12)
avg_annual_cycles = st.number_input("Average Annual Cycles", min_value=0, value=100)
avg_annual_hours = st.number_input("Average Annual Hours", min_value=0, value=2000)
avg_daily_util = st.number_input("Average Daily Utilisation (hours/day)", min_value=0.0, value=8.0)

region = st.selectbox("MRO Region", ["East Asia", "Middle East", "SE Asia", "USA"])

# --- Regression coefficients (from your model)
coefs = {
    'const': 46.2010,
    'age': -4.4149,
    'avg_annual_cycles': 0.0064,
    'avg_annual_hours': -0.00002693,
    'avg_daily_utilisation': 0.3939,
    'age_squared': 0.1345,
    'age_x_util': 0.0010,
    'log_cycles': -1.0003,
    'mro_region_Middle East': -21.4115,
    'mro_region_SE Asia': 2.4409,
    'mro_region_USA': -4.6034
}

# --- Feature engineering
age_squared = age ** 2
age_x_util = age * avg_daily_util
log_cycles = np.log(avg_annual_cycles if avg_annual_cycles > 0 else 0.1)

# --- Region dummy vars
region_dummies = {
    "East Asia": 0,
    "Middle East": coefs['mro_region_Middle East'],
    "SE Asia": coefs['mro_region_SE Asia'],
    "USA": coefs['mro_region_USA']
}

# --- Prediction calculation
y_hat = (
    coefs['const']
    + coefs['age'] * age
    + coefs['avg_annual_cycles'] * avg_annual_cycles
    + coefs['avg_annual_hours'] * avg_annual_hours
    + coefs['avg_daily_utilisation'] * avg_daily_util
    + coefs['age_squared'] * age_squared
    + coefs['age_x_util'] * age_x_util
    + coefs['log_cycles'] * log_cycles
    + region_dummies[region]
)

st.markdown("---")
st.subheader("📅 Predicted MRO Duration:")
st.metric(label="Estimated Turnaround Time", value=f"{y_hat:.1f} days")

st.markdown("""
---
**Model notes:** Based on your regression with aircraft age, region and usage features. 
Region effect is relative to East Asia baseline.
""")
