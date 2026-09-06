import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="MindShield | Burnout Risk Assessment",
    page_icon="🧠",
    layout="wide",
)


@st.cache_resource
def load_model():
  return joblib.load("logistic_regression_mbi_model.pkl")


model = load_model()

# Header Infographic Section
st.title("🧠 MindShield: Burnout & Mental Health Check")
st.caption(
    "AI-Powered Student & Professional Exhaustion Early Warning System"
)
st.markdown("---")

# Sidebar - User Friendly Questionnaire
st.sidebar.header("📋 Quick Assessment")
st.sidebar.write("Answer a few simple questions to evaluate your risk.")

# Easy Inputs instead of raw clinical scores
mood_level = st.sidebar.select_slider(
    "How often do you feel down, depressed, or hopeless lately?",
    options=["Rarely / Never", "Sometimes", "Often", "Almost Always"],
    value="Sometimes",
)

anxiety_level = st.sidebar.select_slider(
    "How frequently do you feel nervous, anxious, or tense?",
    options=["Rarely / Never", "Sometimes", "Often", "Almost Always"],
    value="Sometimes",
)

health_label = st.sidebar.select_slider(
    "How would you rate your overall physical health?",
    options=["Poor", "Fair", "Good", "Very Good", "Excellent"],
    value="Good",
)

stud_h = st.sidebar.number_input(
    "Weekly Work / Study Hours", min_value=0, max_value=100, value=35
)
age = st.sidebar.slider("Age", 18, 65, 22)
sex_label = st.sidebar.selectbox("Gender", options=["Female", "Male"])
job_label = st.sidebar.radio("Are you currently working a job?", ["No", "Yes"])
psyt_label = st.sidebar.radio(
    "Have you ever attended Psychological Therapy?", ["No", "Yes"]
)

# Convert User-Friendly Answers to Model Input Scores
cesd_mapping = {
    "Rarely / Never": 8,
    "Sometimes": 18,
    "Often": 32,
    "Almost Always": 48,
}
stai_mapping = {
    "Rarely / Never": 28,
    "Sometimes": 42,
    "Often": 56,
    "Almost Always": 68,
}
health_mapping = {
    "Poor": 1,
    "Fair": 2,
    "Good": 3,
    "Very Good": 4,
    "Excellent": 5,
}

cesd = cesd_mapping[mood_level]
stai_t = stai_mapping[anxiety_level]
health = health_mapping[health_label]
sex = 0 if sex_label == "Female" else 1
job = 1 if job_label == "Yes" else 0
psyt = 1 if psyt_label == "Yes" else 0

# Base Feature Dictionary
features_dict = {
    "age": age,
    "year": 2,
    "sex": sex,
    "glang": 1,
    "part": 1.0,
    "job": job,
    "stud_h": stud_h,
    "health": health,
    "psyt": psyt,
    "jspe": 105,
    "qcae_cog": 58,
    "qcae_aff": 34,
    "amsp": 23,
    "erec_mean": 0.72,
    "cesd": cesd,
    "stai_t": stai_t,
}

input_df = pd.DataFrame([features_dict])

# Prediction
prob = model.predict_proba(input_df)[0][1]
risk_score = prob * 100

# Top Infographic Dashboard Layout
col1, col2, col3 = st.columns(3)
col1.metric(label="Calculated Mood Stress (CES-D)", value=f"{cesd} / 60")
col2.metric(label="Anxiety Level (STAI-T)", value=f"{stai_t} / 80")
col3.metric(label="Health Index", value=f"{health} / 5")

st.markdown("---")

left_col, right_col = st.columns([1, 1])

# Left Column: Interactive Gauge Meter
with left_col:
  st.subheader("🎯 Burnout Risk Level")

  fig = go.Figure(
      go.Indicator(
          mode="gauge+number",
          value=risk_score,
          number={"suffix": "%"},
          gauge={
              "axis": {"range": [0, 100]},
              "bar": {"color": "#1f77b4"},
              "steps": [
                  {"range": [0, 35], "color": "#2ca02c"},
                  {"range": [35, 65], "color": "#ff7f0e"},
                  {"range": [65, 100], "color": "#d62728"},
              ],
          },
      )
  )
  fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
  st.plotly_chart(fig, use_container_width=True)

  if risk_score >= 65:
    st.error("🚨 **High Exhaustion Risk**: Immediate self-care required.")
  elif risk_score >= 35:
    st.warning(
        "⚠️ **Moderate Exhaustion Risk**: Monitor your stress levels closely."
    )
  else:
    st.success("✅ **Low Exhaustion Risk**: You have a healthy balance!")

# Right Column: What-If Actionable Steps
with right_col:
  st.subheader("💡 What-If Simulator & Action Plan")
  st.write("See how boosting health & managing work hours reduces risk:")

  sim_health = st.slider("Simulate Improving Health Grade", 1, 5, value=health)
  sim_study = st.slider("Simulate Reduced Study/Work Hours", 0, 70, value=stud_h)

  sim_df = input_df.copy()
  sim_df["health"] = sim_health
  sim_df["stud_h"] = sim_study
  sim_prob = model.predict_proba(sim_df)[0][1] * 100

  diff = sim_prob - risk_score
  st.metric(
      label="Projected Risk Score",
      value=f"{sim_prob:.1f}%",
      delta=f"{diff:.1f}%",
  )

  st.markdown("### 📌 Personalized Recommendations")
  if cesd > 20:
    st.markdown("- 🧘 **Take Mental Breaks**: High depressive symptoms noted.")
  if stud_h > 40:
    st.markdown("- ⏰ **Cap Working Hours**: Over 40 hours increases fatigue.")
  if health <= 2:
    st.markdown(
        "- 🏃 **Improve Physical Wellness**: Light daily activity builds"
        " resilience."
    )
