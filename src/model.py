import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from dataset import GREEN, SAFETY_CAR, build_modelling_set, load_raw  # noqa: E402

FEATURES = [
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "IsSoft",
    "IsMedium",
    "IsHard",
    "IsSafetyCar",
    "PaceVsRaceMedian",
    "PaceVsDriverBaseline",
    "PaceTrend3",
]

TARGET = "PitInNext3Laps"
GROUP = "Event"

RF_PARAMS = dict(n_estimators=300, max_depth=8,
                 random_state=42, class_weight="balanced")
THRESHOLD_GRID = np.arange(0.05, 0.95, 0.01)
PRECISION_FLOOR = 0.40


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
        m = RandomForestClassifier(**RF_PARAMS).fit(X.iloc[tr], y.iloc[tr])
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


def run():
    raw = load_raw()
    from dataset import classify_pit_context

    raw = classify_pit_context(raw)
    df = prepare(build_modelling_set(raw))
    df["TargetContext"] = target_context(df, raw)

    X, y, groups = df[FEATURES], df[TARGET], df[GROUP]
    folds, rows = GroupKFold(n_splits=groups.nunique()), []

    for tr, te in folds.split(X, y, groups):
        held = groups.iloc[te].iloc[0]
        Xtr, ytr, Xte, yte = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]

        t = pick_threshold(None, Xtr, ytr, groups.iloc[tr])
        model = RandomForestClassifier(**RF_PARAMS).fit(Xtr, ytr)
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
        )

        ctx = df["TargetContext"].iloc[te]
        for name, label in (("green", GREEN), ("sc", SAFETY_CAR)):
            mask = (ctx == label) | (yte == 0)
            if (yte[mask].sum()) > 0:
                rec[f"recall_{name}"] = (
                    preds[mask.values][yte[mask].values == 1].mean()
                )
        rows.append(rec)

    res = pd.DataFrame(rows)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "\n")

    print("across held-out races (mean +/- std):")
    for m in ("precision", "recall", "f1", "pr_auc", "accuracy"):
        print(f"  {m:10s} {res[m].mean():.3f} +/- {res[m].std():.3f}")
    for m in ("recall_green", "recall_sc"):
        if m in res:
            print(f"  {m:10s} {res[m].mean():.3f} +/- {res[m].std():.3f}")

    print("\nbaselines on the full set:")
    for name, vals in baselines(y, X).items():
        print(f"  {name:16s} " +
              "  ".join(f"{k}={v:.3f}" for k, v in vals.items()))


if __name__ == "__main__":
    run()
