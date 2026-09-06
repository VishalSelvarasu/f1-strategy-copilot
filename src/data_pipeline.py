import fastf1
import pandas as pd
from pathlib import Path


cache_dir = Path("cache")
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)


RACES = [
    (2024, "Bahrain", "R"),
    (2024, "Saudi Arabian Grand Prix", "R"),
    (2024, "Australian Grand Prix", "R"),
    (2024, "Japanese Grand Prix", "R"),
    (2024, "Chinese Grand Prix", "R"),
]


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
    print(f"\nLoading {season} - {event} - {session_code}")
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
    df["Season"] = season
    df["Event"] = event

    df = add_next_pit_label(df, window=3)
    df = add_engineered_features(df)

    return df


def build_dataset(races):
    all_races = []

    for season, event, session_code in races:
        try:
            race_df = download_race(season, event, session_code)
            all_races.append(race_df)
            print(f"Added {event}: {len(race_df)} rows")
        except Exception as e:
            print(f"Failed for {event}: {e}")

    final_df = pd.concat(all_races, ignore_index=True)
    return final_df


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)

    df = build_dataset(RACES)
    df.to_csv("data/race_data.csv", index=False)

    print("\nDataset shape:", df.shape)
    print("\nLabel counts:")
    print(df[["WillPitThisLap", "NextPitIn3Laps"]].sum())

    print("\nRaces included:")
    print(df["Event"].value_counts())

    print("\nSaved to data/race_data.csv")