import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from dataset import GREEN, SAFETY_CAR, build_modelling_set, load_raw  # noqa: E402

# Absolute lap and sector times encode circuit identity: Australia's median lap
# (82.7s) sits below China's fastest (97.8s), so lap-time splits learned in
# training fire on the wrong side of the distribution when a circuit is held
# out. Replaced with circuit-invariant pace deltas computed in dataset.py.
FEATURES = [
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "PaceVsRaceMedian",
    "PaceVsDriverBaseline",
    "PaceTrend3",
    "IsSoft",
    "IsMedium",
    "IsHard",
    "IsSafetyCar",
]

TARGET = "PitInNext3Laps"
GROUP = "Event"

RF_PARAMS = dict(n_estimators=300, max_depth=8,
                 random_state=42, class_weight="balanced")
THRESHOLD_GRID = np.arange(0.05, 0.95, 0.01)

OOF_PATH = Path("data/oof_predictions.csv")
CALIB_PATH = Path("docs/calibration.png")

# Isotonic needs more data than we have per fold; sigmoid is the safer choice
# on ~3,700 training rows with ~400 positives.
CALIBRATION_METHOD = "sigmoid"

DISPLAY_COLS = [
    "Season", "Event", "Driver", "Team", "LapNumber", "Stint", "TyreLife",
    "Position", "Compound", "LapTimeSeconds", "PaceVsRaceMedian",
    "PaceVsDriverBaseline", "PaceTrend3", "IsSafetyCar", "TargetContext",
]


def prepare(df):
    # FastF1 track status 4 is Safety Car, not yellow. The raw pipeline named
    # this column wrong; correcting it here until it moves into dataset.py.
    out = df.rename(columns={"TrackStatusYellow": "IsSafetyCar"}).copy()
    return out.dropna(subset=FEATURES)


def target_context(df, raw, window=3):
    """For each positive row, which kind of pit does its window point at?"""
    keys = ["Season", "Event", "Driver"]
    pits = raw[raw["WillPitThisLap"] == 1]
    lookup = {
        (s, e, d, lap): ctx
        for (s, e, d, lap, ctx) in pits[keys + ["LapNumber", "PitContext"]].itertuples(
            index=False, name=None
        )
    } if "PitContext" in pits.columns else {}

    ctx = []
    for row in df[keys + ["LapNumber", TARGET]].itertuples(index=False, name=None):
        s, e, d, lap, y = row
        if not y:
            ctx.append(None)
            continue
        hit = next(
            (lookup.get((s, e, d, lap + k)) for k in range(1, window + 1)
             if lookup.get((s, e, d, lap + k))),
            None,
        )
        ctx.append(hit)
    return pd.Series(ctx, index=df.index)


def pick_threshold(model, X, y, groups):
    """Choose the F1-optimal threshold using races inside the training fold."""
    probs = np.zeros(len(y))
    inner = GroupKFold(n_splits=min(4, groups.nunique()))
    for tr, va in inner.split(X, y, groups):
        m = fit_calibrated(X.iloc[tr], y.iloc[tr], groups.iloc[tr])
        probs[va] = m.predict_proba(X.iloc[va])[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        _, _, f1, _ = precision_recall_fscore_support(
            y, (probs >= t).astype(int), average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def baselines(y_true, X_test):
    """Is the forest beating the obvious?"""
    out = {}
    _, _, f1, _ = precision_recall_fscore_support(
        y_true, np.zeros(len(y_true), int), average="binary", zero_division=0
    )
    out["always_no_pit"] = dict(
        precision=0.0, recall=0.0, f1=f1, pr_auc=y_true.mean())

    tyre = (X_test["TyreLife"] >=
            X_test["TyreLife"].quantile(0.80)).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, tyre, average="binary", zero_division=0
    )
    out["tyre_life_p80"] = dict(
        precision=p, recall=r, f1=f, pr_auc=average_precision_score(
            y_true, X_test["TyreLife"])
    )
    return out


def fit_calibrated(X, y, groups):
    """Random Forest with probabilities calibrated on held-out races.

    CalibratedClassifierCV's default cv would split rows at random, which is
    the leak this project spent its evaluation design avoiding. Passing an
    explicit GroupKFold keeps whole races on one side of the calibration split.
    """
    inner = GroupKFold(n_splits=min(4, groups.nunique()))
    model = CalibratedClassifierCV(
        RandomForestClassifier(**RF_PARAMS),
        method=CALIBRATION_METHOD,
        cv=list(inner.split(X, y, groups)),
    )
    return model.fit(X, y)


def write_calibration_plot(frames):
    """Reliability curve: does a predicted 0.7 mean 0.7?"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping calibration plot")
        return

    oof = pd.concat(frames, ignore_index=True)
    oof = oof[oof["PitProbability"].notna()]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="#888", label="perfect")
    frac, mean_pred = calibration_curve(
        oof["ActualPitInNext3"], oof["PitProbability"], n_bins=10, strategy="quantile"
    )
    ax.plot(mean_pred, frac, "o-", color="#4c78a8", label="out-of-fold")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed pit rate")
    ax.set_title("Reliability, pooled across held-out races")
    ax.legend()
    fig.tight_layout()

    CALIB_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(CALIB_PATH, dpi=140)
    plt.close(fig)
    print(f"wrote {CALIB_PATH}")


def write_oof(frames, raw):
    """Out-of-fold predictions: every lap scored by a model that never saw its race.

    In-laps are absent from the modelling set, so actual stop laps are joined
    back from the raw frame for display. Those rows carry no prediction.
    """
    oof = pd.concat(frames, ignore_index=True)

    pits = raw.loc[raw["WillPitThisLap"] == 1,
                   ["Season", "Event", "Driver", "LapNumber", "PitContext"]].copy()
    pits = pits.rename(columns={"PitContext": "ActualPitContext"})
    pits["ActualPitLap"] = 1

    oof = oof.merge(pits, on=["Season", "Event",
                    "Driver", "LapNumber"], how="outer")
    oof["ActualPitLap"] = oof["ActualPitLap"].fillna(0).astype(int)

    OOF_PATH.parent.mkdir(exist_ok=True)
    oof.sort_values(["Event", "Driver", "LapNumber"]
                    ).to_csv(OOF_PATH, index=False)
    print(f"\nwrote {OOF_PATH}: {len(oof)} rows, "
          f"{int(oof['ActualPitLap'].sum())} actual stops marked")


def run():
    raw = load_raw()
    from dataset import classify_pit_context

    raw = classify_pit_context(raw)
    df = prepare(build_modelling_set(raw))
    df["TargetContext"] = target_context(df, raw)

    X, y, groups = df[FEATURES], df[TARGET], df[GROUP]
    folds, rows, oof_frames = GroupKFold(n_splits=groups.nunique()), [], []

    for tr, te in folds.split(X, y, groups):
        held = groups.iloc[te].iloc[0]
        Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]

        t = pick_threshold(None, Xtr, ytr, groups.iloc[tr])
        model = fit_calibrated(Xtr, ytr, groups.iloc[tr])
        probs = model.predict_proba(Xte)[:, 1]
        preds = (probs >= t).astype(int)

        p, r, f1, _ = precision_recall_fscore_support(
            yte, preds, average="binary", zero_division=0
        )
        rec = dict(
            race=held, threshold=round(t, 2), n=len(yte), pos=int(yte.sum()),
            precision=p, recall=r, f1=f1, pr_auc=average_precision_score(
                yte, probs),
            accuracy=(preds == yte).mean(),
            brier=brier_score_loss(yte, probs),
        )

        ctx = df["TargetContext"].iloc[te]
        for name, label in (("green", GREEN), ("sc", SAFETY_CAR)):
            mask = (ctx == label) | (yte == 0)
            if (yte[mask].sum()) > 0:
                rec[f"recall_{name}"] = (
                    preds[mask.values][yte[mask].values == 1].mean()
                )
        fold_out = df.iloc[te][[
            c for c in DISPLAY_COLS if c in df.columns]].copy()
        fold_out["ActualPitInNext3"] = yte.values
        fold_out["PitProbability"] = probs
        fold_out["PitPredicted"] = preds
        fold_out["Threshold"] = t
        oof_frames.append(fold_out)

        rows.append(rec)

    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "\n")

    print("across held-out races (mean +/- std):")
    for m in ("precision", "recall", "f1", "pr_auc", "brier", "accuracy"):
        print(f"  {m:10s} {res[m].mean():.3f} +/- {res[m].std():.3f}")
    for m in ("recall_green", "recall_sc"):
        if m in res:
            print(f"  {m:10s} {res[m].mean():.3f} +/- {res[m].std():.3f}")

    print("\nbaselines on the full set:")
    for name, vals in baselines(y, X).items():
        print(f"  {name:16s} " +
              "  ".join(f"{k}={v:.3f}" for k, v in vals.items()))

    write_calibration_plot(oof_frames)
    write_oof(oof_frames, raw)


if __name__ == "__main__":
    run()
