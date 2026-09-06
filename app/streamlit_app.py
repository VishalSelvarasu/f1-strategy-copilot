import pandas as pd
import streamlit as st

st.set_page_config(page_title="F1 Strategy Copilot", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("data/model_predictions.csv")

def explain_row(row, threshold=0.60):
    reasons = []

    if row["TyreLife"] >= 18:
        reasons.append("high tyre age")
    if row["LapTimeSeconds"] >= 95:
        reasons.append("lap pace has dropped")
    if row["Stint"] >= 2:
        reasons.append("later stint phase")
    if row["PredictedProbability"] >= threshold:
        reasons.append("model confidence is above threshold")

    if not reasons:
        return "Pit risk remains moderate; no strong trigger detected."

    return "Pit window risk is elevated because " + ", ".join(reasons) + "."

df = load_data()

st.title("F1 Strategy Copilot")
st.write("Pit-window prediction dashboard based on FastF1 multi-race lap data.")

events = sorted(df["Event"].dropna().unique())
selected_event = st.selectbox("Select race", events)

event_df = df[df["Event"] == selected_event].copy()

drivers = sorted(event_df["Driver"].dropna().unique())
selected_driver = st.selectbox("Select driver", drivers)

driver_df = event_df[event_df["Driver"] == selected_driver].copy()
driver_df = driver_df.sort_values("LapNumber")

st.subheader("Driver race view")
st.dataframe(driver_df[[
    "Season", "Event", "Driver", "LapNumber", "Compound", "TyreLife",
    "Stint", "Position", "LapTimeSeconds",
    "PredictedProbability", "PredictedNextPitIn3Laps",
    "ActualNextPitIn3Laps"
]])

st.subheader("High-risk pit window laps")
high_risk_df = driver_df[driver_df["PredictedProbability"] >= 0.60].copy()

if high_risk_df.empty:
    st.info("No high-risk pit window laps found for this driver at the current threshold.")
else:
    high_risk_df["Explanation"] = high_risk_df.apply(explain_row, axis=1)
    st.dataframe(high_risk_df[[
        "LapNumber", "Compound", "TyreLife", "Stint", "Position",
        "LapTimeSeconds", "PredictedProbability", "Explanation"
    ]])

st.subheader("Top 10 highest pit probabilities in this race")
top_risk = event_df.sort_values("PredictedProbability", ascending=False).head(10)
st.dataframe(top_risk[[
    "Driver", "LapNumber", "Compound", "TyreLife", "Stint",
    "Position", "PredictedProbability",
    "PredictedNextPitIn3Laps", "ActualNextPitIn3Laps"
]])