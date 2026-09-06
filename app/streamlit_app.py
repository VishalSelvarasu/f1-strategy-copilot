from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

OOF_PATH = Path("data/oof_predictions.csv")

CONTEXT_COLOUR = {
    "GREEN": "#2e9e4f",
    "SAFETY_CAR": "#e8b923",
    "RED_FLAG": "#c0392b",
}


@st.cache_data
def load():
    if not OOF_PATH.exists():
        return None
    df = pd.read_csv(OOF_PATH)
    df["LapNumber"] = df["LapNumber"].astype(int)
    return df


st.set_page_config(page_title="F1 Strategy Copilot", layout="wide")
st.title("F1 Strategy Copilot")
st.caption(
    "Probability a driver pits on the next three laps, using only information "
    "available at the end of the current lap. Every prediction shown is "
    "out-of-fold: the model that scored each race was trained on the other four."
)

df = load()
if df is None:
    st.error(f"{OOF_PATH} not found. Run `python src/model.py` first.")
    st.stop()

race = st.sidebar.selectbox("Race", sorted(df["Event"].unique()))
race_df = df[df["Event"] == race]

drivers = sorted(race_df["Driver"].dropna().unique())
driver = st.sidebar.selectbox("Driver", drivers)

threshold = race_df["Threshold"].dropna().iloc[0]
st.sidebar.metric("Fold threshold", f"{threshold:.2f}")
st.sidebar.caption(
    "Selected inside the training folds for this race, never on the race itself."
)

# ---------------------------------------------------------------- race context

stops = race_df[race_df["ActualPitLap"] == 1]
counts = stops["ActualPitContext"].value_counts()

st.subheader(f"{race} — pit events by track state")
c1, c2, c3 = st.columns(3)
c1.metric("Green flag", int(counts.get("GREEN", 0)))
c2.metric("Safety car", int(counts.get("SAFETY_CAR", 0)))
c3.metric("Red flag", int(counts.get("RED_FLAG", 0)))

if counts.get("SAFETY_CAR", 0) > counts.get("GREEN", 0):
    st.warning(
        "Most stops in this race happened under a safety car. Those are not "
        "strategy calls in the usual sense — the pit loss collapses under "
        "neutralisation, so the decision is driven by track state rather than "
        "tyre condition. Prediction quality here is expected to be poor."
    )

# ------------------------------------------------------------- driver timeline

d = race_df[race_df["Driver"] == driver].sort_values("LapNumber")
scored = d[d["PitProbability"].notna()]

st.subheader(f"{driver} — lap-by-lap pit window probability")

if scored.empty:
    st.info("No scored laps for this driver.")
else:
    line = (
        alt.Chart(scored)
        .mark_line(color="#4c78a8")
        .encode(
            x=alt.X("LapNumber:Q", title="Lap"),
            y=alt.Y("PitProbability:Q", title="P(pit in next 3 laps)",
                    scale=alt.Scale(domain=[0, 1])),
            tooltip=["LapNumber", "PitProbability", "TyreLife", "Compound",
                     "Position", "PaceVsDriverBaseline"],
        )
    )

    rule = (
        alt.Chart(pd.DataFrame({"t": [threshold]}))
        .mark_rule(color="#888", strokeDash=[4, 4])
        .encode(y="t:Q")
    )

    layers = [line, rule]

    truth = scored[scored["ActualPitInNext3"] == 1]
    if not truth.empty:
        layers.append(
            alt.Chart(truth)
            .mark_point(color="#4c78a8", size=45, opacity=0.5)
            .encode(x="LapNumber:Q", y="PitProbability:Q")
        )

    actual = d[d["ActualPitLap"] == 1]
    if not actual.empty:
        layers.append(
            alt.Chart(actual)
            .mark_rule(strokeWidth=2)
            .encode(
                x="LapNumber:Q",
                color=alt.Color(
                    "ActualPitContext:N",
                    scale=alt.Scale(
                        domain=list(CONTEXT_COLOUR),
                        range=list(CONTEXT_COLOUR.values()),
                    ),
                    legend=alt.Legend(title="Actual stop"),
                ),
                tooltip=["LapNumber", "ActualPitContext"],
            )
        )

    st.altair_chart(alt.layer(*layers).properties(height=340),
                    use_container_width=True)
    st.caption(
        "Dashed line is the operating threshold. Faint points mark laps that "
        "really were inside a pit window. Vertical rules mark the laps the "
        "driver actually entered the pits, coloured by track state; those laps "
        "are excluded from the model, because a pit-in lap time contains the "
        "pit entry itself."
    )

# ----------------------------------------------------------------- flagged laps

flagged = scored[scored["PitPredicted"] == 1]
st.subheader(f"Laps flagged for {driver} ({len(flagged)} of {len(scored)})")

if flagged.empty:
    st.info("No laps crossed the threshold for this driver.")
else:
    show = flagged[["LapNumber", "PitProbability", "ActualPitInNext3", "Stint",
                    "TyreLife", "Compound", "Position", "PaceVsDriverBaseline"]]
    st.dataframe(
        show.rename(columns={
            "PitProbability": "P(pit)",
            "ActualPitInNext3": "Was a window",
            "PaceVsDriverBaseline": "Pace vs own median (s)",
        }).round(3),
        use_container_width=True,
        hide_index=True,
    )
    hits = int(flagged["ActualPitInNext3"].sum())
    st.caption(
        f"{hits} of {len(flagged)} flagged laps were real pit windows "
        f"({hits / len(flagged):.0%} precision for this driver). Across all "
        "held-out races precision averages 0.33, so roughly two in three alerts "
        "are false. That is the cost of catching most real windows at this "
        "threshold."
    )

# -------------------------------------------------------------- race-wide view

st.subheader(f"Highest-probability laps in {race}")
top = (
    race_df[race_df["PitProbability"].notna()]
    .nlargest(15, "PitProbability")[
        ["Driver", "LapNumber", "PitProbability", "ActualPitInNext3",
         "TyreLife", "Compound", "Position"]
    ]
)
st.dataframe(
    top.rename(columns={"PitProbability": "P(pit)",
               "ActualPitInNext3": "Was a window"}).round(3),
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.caption(
    "This is a behavioural prediction model, not a strategy optimiser. A high "
    "probability means drivers in comparable race states historically pitted "
    "soon — not that stopping is the right call. See the README for evaluation "
    "method and limitations."
)
