import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_weather_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from raw NASA-weather records."""
    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    # Make daily index continuous and fill safely for time series modeling.
    df = df.asfreq("D")
    df = df.interpolate(method="time").ffill().bfill()

    # Temperature target: daily average from max/min temperature.
    df["T2M_AVG"] = (df["T2M_MAX"] + df["T2M_MIN"]) / 2

    # Rainfall targets.
    df["rain_status_target"] = (df["PRECTOTCORR"] > 0).astype(int)
    df["rain_amount_log"] = np.log1p(df["PRECTOTCORR"])

    # Seasonal calendar features.
    df["dayofyear"] = df.index.dayofyear
    df["month"] = df.index.month
    for k in range(1, 4):
        df[f"sin365_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / 365.25)
        df[f"cos365_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / 365.25)

    # Change features: important before rainfall events.
    for col in ["RH2M", "PS", "T2MDEW", "WS2M", "T2M_AVG"]:
        df[f"{col}_change1"] = df[col].diff()
        df[f"{col}_change3"] = df[col].diff(3)

    # Lag and rolling features for meteorological memory.
    lag_base_cols = ["PRECTOTCORR", "RH2M", "T2MDEW", "WS2M", "PS", "T2M_AVG"]
    for col in lag_base_cols:
        for lag in [1, 3, 7, 14, 30]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for window in [3, 7, 14, 30]:
            df[f"{col}_rollmean{window}"] = df[col].shift(1).rolling(window).mean()
            df[f"{col}_rollstd{window}"] = df[col].shift(1).rolling(window).std()

    return df.dropna()


def load_and_prepare(path: str) -> pd.DataFrame:
    """Load NASA weather data and create weather-memory features."""
    return prepare_weather_dataframe(pd.read_csv(path))


def create_sequences(X, y_temp, y_status, y_amount, window: int):
    X_seq, temp_seq, status_seq, amount_seq = [], [], [], []
    for i in range(window, len(X)):
        X_seq.append(X[i-window:i])
        temp_seq.append(y_temp[i])
        status_seq.append(y_status[i])
        amount_seq.append(y_amount[i])
    return (
        np.array(X_seq, dtype=np.float32),
        np.array(temp_seq, dtype=np.float32),
        np.array(status_seq, dtype=np.float32),
        np.array(amount_seq, dtype=np.float32),
    )


def build_dataset(df: pd.DataFrame, window: int, val_size: float, test_size: float):
    exclude = ["PRECTOTCORR", "RainStatus", "rain_status_target", "rain_amount_log"]
    feature_cols = [c for c in df.columns if c not in exclude]

    feature_scaler = MinMaxScaler()
    temp_scaler = MinMaxScaler()
    amount_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(df[feature_cols])
    y_temp = temp_scaler.fit_transform(df[["T2M_AVG"]]).reshape(-1)
    y_status = df["rain_status_target"].values
    y_amount = amount_scaler.fit_transform(df[["rain_amount_log"]]).reshape(-1)

    X, y_temp_seq, y_status_seq, y_amount_seq = create_sequences(
        X_scaled, y_temp, y_status, y_amount, window
    )

    n = len(X)
    train_end = int(n * (1 - val_size - test_size))
    val_end = int(n * (1 - test_size))

    return {
        "X_train": X[:train_end],
        "X_val": X[train_end:val_end],
        "X_test": X[val_end:],
        "y_temp_train": y_temp_seq[:train_end],
        "y_temp_val": y_temp_seq[train_end:val_end],
        "y_temp_test": y_temp_seq[val_end:],
        "y_status_train": y_status_seq[:train_end],
        "y_status_val": y_status_seq[train_end:val_end],
        "y_status_test": y_status_seq[val_end:],
        "y_amount_train": y_amount_seq[:train_end],
        "y_amount_val": y_amount_seq[train_end:val_end],
        "y_amount_test": y_amount_seq[val_end:],
        "feature_cols": feature_cols,
        "feature_scaler": feature_scaler,
        "temp_scaler": temp_scaler,
        "amount_scaler": amount_scaler,
    }
