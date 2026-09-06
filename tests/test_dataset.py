import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import (  # noqa: E402
    GREEN,
    RED_FLAG,
    SAFETY_CAR,
    build_modelling_set,
    classify_pit_context,
    relabel,
)


def make_race(event="Test GP", driver="XXX", laps=20, pit_laps=(), status=None,
              season=2024):
    """One driver, one race, pits on the given laps."""
    status = status or {}
    return pd.DataFrame({
        "Season": season,
        "Event": event,
        "Driver": driver,
        "LapNumber": [float(n) for n in range(1, laps + 1)],
        "WillPitThisLap": [1 if n in pit_laps else 0 for n in range(1, laps + 1)],
        "TrackStatus": [str(status.get(n, "1")) for n in range(1, laps + 1)],
        "LapTimeSeconds": [90.0] * laps,
    })


# --------------------------------------------------------------- label horizon

def test_label_excludes_current_lap():
    """A pit on lap 10 must not make lap 10 itself positive.

    The in-lap contains the pit entry in its own lap time (median +5.4s over
    the driver's race median, up to +45s). Including it in its own target
    window leaks that signal straight into the features.
    """
    df = relabel(make_race(pit_laps={10}))
    positives = set(df.loc[df["PitInNext3Laps"] == 1, "LapNumber"])
    assert positives == {7.0, 8.0, 9.0}
    assert 10.0 not in positives


def test_label_window_size_is_respected():
    df = relabel(make_race(pit_laps={10}), window=5)
    assert set(df.loc[df["PitInNext3Laps"] == 1, "LapNumber"]) == {
        5.0, 6.0, 7.0, 8.0, 9.0}


def test_label_survives_missing_laps():
    """Labels key on lap number, not row position.

    The raw pipeline drops rows with missing timing data, so row i+1 is not
    always lap n+1. Positional slicing would shift the window across a gap.
    """
    df = make_race(pit_laps={10})
    df = df[df["LapNumber"] != 8.0]  # drop a lap inside the window
    out = relabel(df)
    positives = set(out.loc[out["PitInNext3Laps"] == 1, "LapNumber"])
    assert positives == {7.0, 9.0}
    assert 6.0 not in positives  # would appear if the window slid by one row


def test_no_positives_when_no_pits():
    df = relabel(make_race(pit_laps=set()))
    assert df["PitInNext3Laps"].sum() == 0


def test_window_running_off_the_start_is_safe():
    """A pit on lap 2 points at laps -1, 0, 1 -- only lap 1 exists."""
    df = relabel(make_race(pit_laps={2}))
    assert set(df.loc[df["PitInNext3Laps"] == 1, "LapNumber"]) == {1.0}


# ------------------------------------------------------------------- grouping

def test_labels_do_not_cross_races():
    """A pit early in race B must not label the final laps of race A.

    add_next_pit_label in data_pipeline.py groups by Driver alone, which is
    only safe because it runs per race. This function groups by
    (Season, Event, Driver) so it stays correct on a concatenated frame.
    """
    a = make_race(event="Race A", laps=10, pit_laps=set())
    b = make_race(event="Race B", laps=10, pit_laps={1})
    out = relabel(pd.concat([a, b], ignore_index=True))
    assert out.loc[out["Event"] == "Race A", "PitInNext3Laps"].sum() == 0


def test_labels_do_not_cross_drivers():
    a = make_race(driver="AAA", laps=10, pit_laps=set())
    b = make_race(driver="BBB", laps=10, pit_laps={1})
    out = relabel(pd.concat([a, b], ignore_index=True))
    assert out.loc[out["Driver"] == "AAA", "PitInNext3Laps"].sum() == 0


def test_labels_do_not_cross_seasons():
    a = make_race(season=2024, laps=10, pit_laps=set())
    b = make_race(season=2025, laps=10, pit_laps={1})
    out = relabel(pd.concat([a, b], ignore_index=True))
    assert out.loc[out["Season"] == 2024, "PitInNext3Laps"].sum() == 0


# ------------------------------------------------------------- pit classification

def test_lone_green_flag_stop_is_green():
    df = classify_pit_context(make_race(pit_laps={10}))
    assert df.loc[df["LapNumber"] == 10.0, "PitContext"].iloc[0] == GREEN


def test_safety_car_status_is_tagged():
    df = classify_pit_context(make_race(pit_laps={10}, status={10: "4"}))
    assert df.loc[df["LapNumber"] == 10.0, "PitContext"].iloc[0] == SAFETY_CAR


def test_same_lap_cluster_is_a_neutralisation():
    """Fourteen cars stopping on one lap is a safety car, not fourteen calls.

    This is Saudi Arabia 2024 lap 7: 14 of the race's 20 pit events.
    """
    field = pd.concat(
        [make_race(driver=f"D{i:02d}", laps=20, pit_laps={7})
         for i in range(14)],
        ignore_index=True,
    )
    out = classify_pit_context(field)
    stops = out[out["WillPitThisLap"] == 1]
    assert (stops["PitContext"] == SAFETY_CAR).all()


def test_lap_one_cluster_is_a_red_flag():
    """Japan 2024: 18 stops on lap 1 under the red flag."""
    field = pd.concat(
        [make_race(driver=f"D{i:02d}", laps=20, pit_laps={1})
         for i in range(18)],
        ignore_index=True,
    )
    out = classify_pit_context(field)
    stops = out[out["WillPitThisLap"] == 1]
    assert (stops["PitContext"] == RED_FLAG).all()


def test_non_pit_laps_have_no_context():
    df = classify_pit_context(make_race(pit_laps={10}))
    assert (df.loc[df["LapNumber"] != 10.0, "PitContext"] != GREEN).all()


# ----------------------------------------------------------- modelling set

def test_in_laps_are_dropped():
    out = build_modelling_set(make_race(pit_laps={10}))
    assert 10.0 not in set(out["LapNumber"])


def test_red_flag_stops_are_excluded():
    field = pd.concat(
        [make_race(driver=f"D{i:02d}", laps=20, pit_laps={1})
         for i in range(18)],
        ignore_index=True,
    )
    out = build_modelling_set(field)
    assert out["PitInNext3Laps"].sum() == 0


def test_safety_car_stops_are_kept():
    field = pd.concat(
        [make_race(driver=f"D{i:02d}", laps=20, pit_laps={7})
         for i in range(14)],
        ignore_index=True,
    )
    out = build_modelling_set(field)
    assert out["PitInNext3Laps"].sum() > 0


def test_target_column_is_present():
    out = build_modelling_set(make_race(pit_laps={10}))
    assert "PitInNext3Laps" in out.columns
    assert set(out["PitInNext3Laps"].unique()) <= {0, 1}


# ------------------------------------------------------ committed data schema

DATA = Path(__file__).resolve().parents[1] / "data" / "race_data.csv"


@pytest.mark.skipif(not DATA.exists(), reason="run src/data_pipeline.py first")
def test_committed_dataset_has_required_columns():
    df = pd.read_csv(DATA, nrows=5)
    required = {"Season", "Event", "Driver", "LapNumber", "WillPitThisLap",
                "TrackStatus", "LapTimeSeconds"}
    assert required <= set(df.columns), required - set(df.columns)


@pytest.mark.skipif(not DATA.exists(), reason="run src/data_pipeline.py first")
def test_committed_dataset_has_no_duplicate_laps():
    df = pd.read_csv(DATA)
    key = ["Season", "Event", "Driver", "LapNumber"]
    assert not df.duplicated(subset=key).any()
