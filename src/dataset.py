from pathlib import Path

import pandas as pd

RAW_PATH = Path("data/race_data.csv")

GROUP_KEYS = ["Season", "Event", "Driver"]
RACE_KEYS = ["Season", "Event"]

# FastF1 track status codes. 4 is Safety Car, not yellow -- the raw pipeline's
# "TrackStatusYellow" column is misnamed.
TS_YELLOW = "2"
TS_SAFETY_CAR = "4"
TS_RED_FLAG = "5"
TS_VSC = "6"

# A single-lap cluster this large in a ~20 car field is a neutralisation, not a
# set of independent strategy calls. Stated assumption, not a tuned parameter.
NEUTRALISATION_CLUSTER = 6

RED_FLAG = "RED_FLAG"
SAFETY_CAR = "SAFETY_CAR"
GREEN = "GREEN"
NOT_A_PIT = "NONE"


def add_pace_features(df):
    """Circuit-invariant pace features.

    Raw lap time is useless across circuits -- a forest trained on Shanghai
    splits at values that never occur at Albert Park. These are all differences
    computed within a single race, so the circuit's absolute pace cancels out.

    Assumes df is already sorted by GROUP_KEYS + LapNumber; PaceTrend3 rolls
    over row order.
    """
    out = df.copy()
    out["PaceVsRaceMedian"] = out["LapTimeSeconds"] - out.groupby(RACE_KEYS)[
        "LapTimeSeconds"
    ].transform("median")
    out["PaceVsDriverBaseline"] = out["LapTimeSeconds"] - out.groupby(GROUP_KEYS)[
        "LapTimeSeconds"
    ].transform("median")
    # Costs the first two laps of every driver's race: min_periods=2 needs two
    # laps to form a mean, and .diff() needs a previous mean to subtract.
    out["PaceTrend3"] = out.groupby(GROUP_KEYS)["LapTimeSeconds"].transform(
        lambda s: s.rolling(3, min_periods=2).mean().diff()
    )
    return out


def load_raw(path=RAW_PATH):
    df = pd.read_csv(path)
    df["TrackStatus"] = df["TrackStatus"].astype(str)
    df = df.sort_values(GROUP_KEYS + ["LapNumber"]).reset_index(drop=True)
    return add_pace_features(df)


def relabel(df, window=3, target="PitInNext3Laps"):
    """Label lap n positive if a pit occurs on laps n+1 .. n+window.

    Uses lap numbers rather than row offsets so that gaps left by the raw
    pipeline's dropna do not shift the window.
    """

    def per_driver(group):
        pit_laps = set(group.loc[group["WillPitThisLap"] == 1, "LapNumber"])
        offsets = range(1, window + 1)
        return pd.Series(
            [
                int(any((lap + k) in pit_laps for k in offsets))
                for lap in group["LapNumber"]
            ],
            index=group.index,
        )

    out = df.copy()
    parts = [per_driver(g) for _, g in out.groupby(GROUP_KEYS, sort=False)]
    # concat rather than groupby.apply: with a single group, apply returns a
    # DataFrame instead of a Series and the assignment below fails.
    out[target] = pd.concat(parts).reindex(out.index).astype(int)
    return out


def classify_pit_context(df):
    """Tag each pit event as red flag, safety car, or green-flag strategy.

    Red flag and safety car stops are real events but different decisions: under
    neutralisation the pit loss collapses, so the call is driven by track status
    rather than tyre state. Blending them into one target is why TrackStatus
    carries almost no importance in the current model.
    """
    out = df.copy()
    out["PitContext"] = NOT_A_PIT

    pits = out["WillPitThisLap"] == 1

    cluster = (
        out[pits]
        .groupby(RACE_KEYS + ["LapNumber"])["Driver"]
        .transform("size")
        .reindex(out.index)
        .fillna(0)
    )

    status = out["TrackStatus"]
    red = status.str.contains(TS_RED_FLAG, na=False)
    neutralised = status.str.contains(TS_SAFETY_CAR, na=False) | status.str.contains(
        TS_VSC, na=False
    )
    clustered = cluster >= NEUTRALISATION_CLUSTER

    out.loc[pits, "PitContext"] = GREEN
    out.loc[pits & (neutralised | clustered), "PitContext"] = SAFETY_CAR
    out.loc[
        pits & (red | (clustered & (out["LapNumber"] <= 2))), "PitContext"
    ] = RED_FLAG
    return out


def build_modelling_set(df=None, window=3, drop_in_laps=True, exclude_red_flag=True):
    """Raw frame -> modelling frame.

    In-laps are dropped because their own lap time encodes the pit entry. Red
    flag stops are dropped because they are not decisions. Safety car stops are
    kept and tagged so they can be evaluated as a separate subset.
    """
    if df is None:
        df = load_raw()

    out = classify_pit_context(df)

    if exclude_red_flag:
        red_drivers = out.loc[out["PitContext"] ==
                              RED_FLAG, GROUP_KEYS + ["LapNumber"]]
        out = out.merge(
            red_drivers.assign(_red=1), on=GROUP_KEYS + ["LapNumber"], how="left"
        )
        out = out[out["_red"].isna()].drop(columns="_red")
        out["WillPitThisLap"] = (
            out["WillPitThisLap"] & (out["PitContext"] != RED_FLAG)
        ).astype(int)

    out = relabel(out, window=window)

    if drop_in_laps:
        out = out[out["WillPitThisLap"] == 0]

    return out.reset_index(drop=True)


if __name__ == "__main__":
    raw = load_raw()
    ctx = classify_pit_context(raw)
    print("pit events by context:")
    print(ctx.loc[ctx["WillPitThisLap"] == 1,
          "PitContext"].value_counts(), "\n")

    model_df = build_modelling_set(raw)
    print("rows:", len(model_df), "(raw:", len(raw), ")")
    print("positives:", int(model_df["PitInNext3Laps"].sum()))
    print("positive rate:", round(model_df["PitInNext3Laps"].mean(), 4), "\n")
    print("rows per race:")
    print(model_df["Event"].value_counts())
