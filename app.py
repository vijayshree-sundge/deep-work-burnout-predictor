"""Streamlit app: Deep Work & Burnout Productivity Predictor.

Run:  streamlit run app.py
"""
import json

import pandas as pd
import streamlit as st

from ai_coach import generate_plan
from focus_curve import fmt_hour
from predictor import load_models, predict
from train_model import METRICS_PATH

st.set_page_config(page_title="Deep Work & Burnout Predictor", page_icon="🧠", layout="wide")


@st.cache_resource
def get_bundle():
    return load_models()


bundle = get_bundle()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Today's inputs")
sleep_hours = st.sidebar.slider("Sleep duration (hours)", 3.0, 10.0, 7.0, 0.5)
sleep_quality = st.sidebar.slider("Sleep quality (1-10)", 1, 10, 6)
screen_time = st.sidebar.slider("Screen time (hours/day)", 0.5, 14.0, 6.0, 0.5)
caffeine = st.sidebar.slider("Caffeine (cups of coffee/tea)", 0, 8, 2)
mood = st.sidebar.slider("Morning mood (1-10)", 1, 10, 6)
study_planned = st.sidebar.slider("Study hours planned", 0.0, 12.0, 5.0, 0.5)

st.sidebar.divider()
api_key = st.sidebar.text_input(
    "Gemini API key (optional)",
    type="password",
    help="Get a free key at https://aistudio.google.com/apikey. "
    "You can also set GEMINI_API_KEY in a .env / environment variable.",
)

profile = {
    "sleep_hours": sleep_hours,
    "sleep_quality": sleep_quality,
    "screen_time_hours": screen_time,
    "caffeine_cups": caffeine,
    "morning_mood": mood,
    "study_hours_planned": study_planned,
}

# ------------------------------------------------------------------- main
st.title("🧠 Deep Work & Burnout Predictor")
st.caption(
    "Linear Regression predicts your productivity and burnout risk from your "
    "sleep, screen time, caffeine and mood. Gemini turns it into an energy plan."
)

if st.sidebar.button("Predict my day", type="primary", use_container_width=True):
    result = predict(profile, bundle)
    st.session_state["result"] = result
    st.session_state["profile"] = profile
    with st.spinner("Your AI coach is writing your plan..."):
        st.session_state["plan"] = generate_plan(profile, result, api_key or None)

if "result" not in st.session_state:
    st.info("Set your inputs in the sidebar and click **Predict my day**.")
else:
    result = st.session_state["result"]
    plan, source = st.session_state["plan"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Productivity score", f"{result['productivity']:.0f} / 100")
    c2.metric("Deep work possible", f"{result['deep_work_hours']} hrs")
    c3.metric("Burnout probability", f"{result['burnout']:.0f}%", result["burnout_label"],
              delta_color="off")

    st.progress(int(result["productivity"]), text="Productivity")
    st.progress(int(result["burnout"]), text="Burnout risk")

    st.subheader("Predicted focus through the day")
    chart_df = pd.DataFrame(
        {"Focus score": list(result["curve"].values())},
        index=[fmt_hour(h) for h in result["curve"]],
    )
    st.line_chart(chart_df)

    p_col, s_col = st.columns(2)
    p_col.success("**Peak windows:** " + (", ".join(result["peak_ranges"]) or "none"))
    s_col.warning("**Slump windows:** " + (", ".join(result["slump_ranges"]) or "none"))

    st.subheader("Your personalised energy plan")
    st.markdown(plan)
    st.caption(f"Plan source: {source}")

with st.expander("About the model"):
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        for target, info in m.items():
            st.write(f"**{target}**  -  R² = {info['r2']}, MAE = {info['mae']}")
            st.json(info["coefficients"])
    st.write(
        "The training data is synthetic (see `generate_data.py`) because a real "
        "labelled dataset was not available. Predictions are for learning and "
        "self-reflection, not medical advice."
    )
