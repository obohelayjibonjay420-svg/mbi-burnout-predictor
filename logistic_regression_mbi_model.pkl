%%writefile app.py
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="MBI Burnout Assessment", page_icon="🧠", layout="wide"
)

# Custom CSS for unique UI
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #2b5c8f;
        color: white;
    }
    </style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
  return joblib.load('logistic_regression_mbi_model.pkl')


model = load_model()

st.title("🧠 MBI Exhaustion & Burnout Predictor")
st.caption(
    "Interactive Machine Learning Assessment with Explainable Diagnostics"
)

# Sidebar Inputs
st.sidebar.header("📋 Subject Assessment Data")

cesd = st.sidebar.slider(
    "Depression Score (CES-D)",
    min_value=0,
    max_value=60,
    value=15,
    help="Higher scores reflect increased depressive symptoms.",
)
stai_t = st.sidebar.slider(
    "Trait Anxiety Score (STAI-T)",
    min_value=20,
    max_value=80,
    value=40,
    help="Higher scores reflect trait anxiety levels.",
)
health = st.sidebar.slider(
    "Self-Reported Health Status",
    min_value=1,
    max_value=5,
    value=3,
    help="1 = Poor, 5 = Excellent",
)
part = st.sidebar.number_input(
    "Participation / Commitment Scale (part)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
)
stud_h = st.sidebar.number_input(
    "Weekly Study Hours (stud_h)", min_value=0, max_value=100, value=30
)
age = st.sidebar.slider("Age", 18, 70, 24)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
psyt = st.sidebar.selectbox("Psychological Therapy History (psyt)", options=[0, 1])
job = st.sidebar.selectbox("Working Job Status", options=[0, 1])

# Fill baseline defaults for remaining trained features
features_dict = {
    'age': age,
    'year': 2,
    'sex': sex,
    'glang': 1,
    'part': part,
    'job': job,
    'stud_h': stud_h,
    'health': health,
    'psyt': psyt,
    'jspe': 100,
    'qcae_cog': 50,
    'qcae_aff': 30,
    'amsp': 25,
    'erec_mean': 0.7,
    'cesd': cesd,
    'stai_t': stai_t,
}

input_df = pd.DataFrame([features_dict])

# Prediction Calculations
prob = model.predict_proba(input_df)[0][1]
risk_score = prob * 100

# Main Dashboard Layout
col1, col2 = st.columns([1, 2])

with col1:
  st.subheader("Assessment Verdict")
  if risk_score >= 50:
    st.error(f"⚠️ High Exhaustion Risk: {risk_score:.1f}%")
  else:
    st.success(f"✅ Low Exhaustion Risk: {risk_score:.1f}%")

  st.progress(prob)

  st.markdown("**Key Risk Contributors:**")
  if cesd > 20:
    st.write("• **Depression Score (`cesd`)**: Elevated")
  if stai_t > 45:
    st.write("• **Anxiety Score (`stai_t`)**: Elevated")
  if health <= 2:
    st.write("• **Health Perception**: Low protective factor")

with col2:
  st.subheader("💡 Interactive Recommendations & What-If Simulation")
  st.write(
      "Adjusting key protective features like self-reported health or"
      " addressing anxiety reduces total burnout risk probability in real time."
  )

  # What-If Simulator
  sim_health = st.slider(
      "Simulate Improved Health Score", 1, 5, value=max(health, 4)
  )
  sim_df = input_df.copy()
  sim_df['health'] = sim_health
  sim_prob = model.predict_proba(sim_df)[0][1] * 100

  st.info(
      f"Potential Risk Score with Health Grade {sim_health}:"
      f" **{sim_prob:.1f}%** (Difference: **{sim_prob - risk_score:+.1f}%**)"
  )