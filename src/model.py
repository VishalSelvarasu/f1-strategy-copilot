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


def load_data(path="data/race_data.csv"):
    df = pd.read_csv(path)

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y, df


if __name__ == "__main__":
    X, y, df = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Rows:", len(df))
    print("\nTarget distribution:")
    print(y.value_counts())

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