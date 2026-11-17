import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

NUMERIC = ["amount","hour_of_day","day_of_week","distance_from_home_km",
           "velocity_1h","amount_z_score","merchant_risk_score"]
BINARY = ["is_foreign_transaction","card_present"]
TARGET = "is_fraud"


def _engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if (h >= 22 or h <= 5) else 0)
    df["amount_velocity"] = df["amount"] * df["velocity_1h"]
    df["distance_foreign"] = df["distance_from_home_km"] * df["is_foreign_transaction"]
    return df


def build_features(df: pd.DataFrame, scaler: StandardScaler = None, fit: bool = True):
    df = _engineer(df)
    dummies = pd.get_dummies(df["merchant_category"], prefix="cat", drop_first=False)
    extra_num = ["is_night", "amount_velocity", "distance_foreign"]
    all_num = NUMERIC + extra_num
    X = pd.concat([df[all_num + BINARY].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    X.columns = X.columns.astype(str)
    if fit:
        scaler = StandardScaler()
        X[all_num] = scaler.fit_transform(X[all_num])
    else:
        X[all_num] = scaler.transform(X[all_num])
    return X, scaler


def preprocess(csv_path, test_size: float = 0.2, val_size: float = 0.2, random_state: int = 42):
    """Split into train / validation / test BEFORE any fitting, so nothing leaks.

    - Scaler is fit on the training rows only.
    - SMOTE only touches the training rows.
    - Validation is used to pick the alert threshold.
    - Test is touched once, at the very end, for the final score.
    """
    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[TARGET])
    train_df, val_df = train_test_split(
        train_df, test_size=val_size / (1 - test_size), random_state=random_state,
        stratify=train_df[TARGET])

    X_train, scaler = build_features(train_df, fit=True)
    feature_names = list(X_train.columns)
    X_val = _align(build_features(val_df, scaler=scaler, fit=False)[0], feature_names)
    X_test = _align(build_features(test_df, scaler=scaler, fit=False)[0], feature_names)
    y_train, y_val, y_test = (d[TARGET].values for d in (train_df, val_df, test_df))

    print(f"Split — train {len(y_train):,} | validation {len(y_val):,} | test {len(y_test):,}")
    print(f"Before SMOTE — fraud: {y_train.sum()} / {len(y_train)}")
    X_train_res, y_train_res = SMOTE(random_state=42, k_neighbors=5).fit_resample(X_train.values, y_train)
    print(f"After  SMOTE — fraud: {y_train_res.sum()} / {len(y_train_res)}")
    return dict(X_train=X_train_res, y_train=y_train_res,
                X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
                scaler=scaler, feature_names=feature_names)


def _align(X: pd.DataFrame, feature_names: list) -> np.ndarray:
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    return X[feature_names].values.astype(float)


def preprocess_new(df: pd.DataFrame, scaler: StandardScaler, feature_names: list) -> np.ndarray:
    X, _ = build_features(df, scaler=scaler, fit=False)
    return _align(X, feature_names)
