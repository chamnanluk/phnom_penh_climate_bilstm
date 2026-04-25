"""
Data utilities for SARIMA + BiLSTM residual learning.

The residual BiLSTM uses multivariate weather sequences as input and SARIMA residuals
as the target.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_hybrid_dataframe(
    csv_path: str,
    date_col: str = "Date",
    target_col: str = "T2M",
) -> pd.DataFrame:
    """
    Load and prepare the daily weather dataframe.

    The code is flexible:
    - If T2M exists, it uses T2M as average temperature.
    - If T2M does not exist but T2M_MAX and T2M_MIN exist, it creates T2M.
    """
    df = pd.read_csv(csv_path)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).set_index(date_col)

    df = df.asfreq("D")
    df = df.interpolate(method="time").ffill().bfill()

    if target_col not in df.columns:
        if "T2M_MAX" in df.columns and "T2M_MIN" in df.columns:
            df[target_col] = (df["T2M_MAX"] + df["T2M_MIN"]) / 2
        elif "T2M_AVG" in df.columns:
            df[target_col] = df["T2M_AVG"]
        else:
            raise ValueError(
                f"Cannot find target temperature column '{target_col}'. "
                "Expected T2M, T2M_AVG, or T2M_MAX/T2M_MIN."
            )

    # Calendar seasonality features
    df["dayofyear"] = df.index.dayofyear
    df["month"] = df.index.month
    for k in range(1, 4):
        df[f"sin365_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365.25)
        df[f"cos365_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365.25)

    # Lag and rolling features for common NASA weather variables
    candidate_cols = [
        "PRECTOTCORR", "RH2M", "T2MDEW", "WS2M", "PS",
        "ALLSKY_SFC_SW_DWN", target_col
    ]

    for col in candidate_cols:
        if col not in df.columns:
            continue

        for lag in [1, 3, 7, 14, 30]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

        for window in [3, 7, 14, 30]:
            df[f"{col}_rollmean{window}"] = df[col].shift(1).rolling(window).mean()
            df[f"{col}_rollstd{window}"] = df[col].shift(1).rolling(window).std()

    # Change features
    for col in ["RH2M", "PS", "T2MDEW", "WS2M"]:
        if col in df.columns:
            df[f"{col}_change"] = df[col].diff()

    df = df.dropna()
    return df


def create_residual_sequences(
    X: np.ndarray,
    residuals: np.ndarray,
    sarima_preds: np.ndarray,
    actual_temp: np.ndarray,
    window: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build supervised sequences for residual prediction.

    At time i, model input is X[i-window:i], target is residual[i].
    """
    X_seq, y_resid, y_sarima, y_actual = [], [], [], []

    for i in range(window, len(X)):
        X_seq.append(X[i - window:i])
        y_resid.append(residuals[i])
        y_sarima.append(sarima_preds[i])
        y_actual.append(actual_temp[i])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_resid, dtype=np.float32).reshape(-1, 1),
        np.asarray(y_sarima, dtype=np.float32).reshape(-1, 1),
        np.asarray(y_actual, dtype=np.float32).reshape(-1, 1),
    )


def scale_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train_df[feature_cols])
    X_test = scaler.transform(test_df[feature_cols])
    return X_train, X_test, scaler
