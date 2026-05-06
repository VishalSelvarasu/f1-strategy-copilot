import fastf1
import pandas as pd
from pathlib import Path


cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)


def add_next_pit_label(df, window=3):
    df = df.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    df["WillPitThisLap"] = df["PitInTime"].notna().astype(int)
    df["NextPitIn3Laps"] = 0

    for driver, group in df.groupby("Driver"):
        idx = group.index.tolist()
        will_pit = group["WillPitThisLap"].tolist()

        for i in range(len(idx)):
            future_window = will_pit[i:i + window]
            if sum(future_window) > 0:
                df.loc[idx[i], "NextPitIn3Laps"] = 1

    return df


def add_engineered_features(df):
    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()
    df["Sector1Seconds"] = df["Sector1Time"].dt.total_seconds()
    df["Sector2Seconds"] = df["Sector2Time"].dt.total_seconds()
    df["Sector3Seconds"] = df["Sector3Time"].dt.total_seconds()

    df["IsSoft"] = (df["Compound"] == "SOFT").astype(int)
    df["IsMedium"] = (df["Compound"] == "MEDIUM").astype(int)
    df["IsHard"] = (df["Compound"] == "HARD").astype(int)

    df["TrackStatus"] = df["TrackStatus"].astype(str)
    df["TrackStatusYellow"] = df["TrackStatus"].str.contains("4", na=False).astype(int)

    df = df.dropna(subset=["LapTimeSeconds", "TyreLife", "Position"])

    return df


def download_race(season, event, session_code="R"):
    session = fastf1.get_session(season, event, session_code)
    session.load()

    laps = session.laps.copy()

    cols = [
        "LapNumber",
        "Driver",
        "Team",
        "Compound",
        "Stint",
        "TyreLife",
        "LapTime",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
        "Position",
        "TrackStatus",
        "Time",
        "DriverNumber",
        "PitInTime",
        "PitOutTime"
    ]

    df = laps[cols].copy()
    df = add_next_pit_label(df, window=3)
    df = add_engineered_features(df)

    return df


if __name__ == "__main__":
    df = download_race(2024, "Bahrain", "R")
    Path("data").mkdir(exist_ok=True)
    df.to_csv("data/race_data.csv", index=False)

    print(df[[
        "LapNumber", "Driver", "Compound", "Stint",
        "PitInTime", "WillPitThisLap", "NextPitIn3Laps",
        "LapTimeSeconds", "TyreLife", "Position"
    ]].head(20))

    print("\nLabel counts:")
    print(df[["WillPitThisLap", "NextPitIn3Laps"]].sum())

    print("\nSaved to data/race_data.csv")