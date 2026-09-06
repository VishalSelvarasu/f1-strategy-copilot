import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


FEATURES = [
    "LapNumber",
    "Stint",
    "TyreLife",
    "Position",
    "LapTimeSeconds",
    "Sector1Seconds",
    "Sector2Seconds",
    "Sector3Seconds",
    "IsSoft",
    "IsMedium",
    "IsHard",
    "TrackStatusYellow"
]

TARGET = "NextPitIn3Laps"
THRESHOLD = 0.60


def load_data(path="data/race_data.csv"):
    df = pd.read_csv(path)
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    return X, y, df


if __name__ == "__main__":
    X, y, df = load_data()

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y,
        df,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= THRESHOLD).astype(int)

    print("Rows:", len(df))
    print("\nTarget distribution:")
    print(y.value_counts())

    print(f"\nThreshold used: {THRESHOLD}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    importances = pd.Series(
        model.feature_importances_,
        index=FEATURES
    ).sort_values(ascending=False)

    print("\nFeature Importances:")
    print(importances)

    results = df_test.copy()
    results["ActualNextPitIn3Laps"] = y_test.values
    results["PredictedProbability"] = probs
    results["PredictedNextPitIn3Laps"] = preds

    results = results.sort_values("PredictedProbability", ascending=False)

    print("\nTop 15 highest pit-window probabilities:")
    print(results[[
        "Season", "Event", "Driver", "LapNumber", "Compound", "TyreLife",
        "Stint", "Position", "PredictedProbability",
        "ActualNextPitIn3Laps", "PredictedNextPitIn3Laps"
    ]].head(15))

    results.to_csv("data/model_predictions.csv", index=False)
    print("\nSaved predictions to data/model_predictions.csv")