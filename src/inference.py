"""Inference helpers for the deployable baseline weather-forecast app."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data import prepare_weather_dataframe


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best_baseline.keras"
ARTIFACTS_PATH = PROJECT_ROOT / "models" / "baseline_artifacts.joblib"


def model_is_ready() -> bool:
    """Return whether the baseline model and its preprocessing artifacts exist."""
    return MODEL_PATH.is_file() and ARTIFACTS_PATH.is_file()


def predict_next_day(raw_weather: pd.DataFrame) -> dict[str, float | bool | str]:
    """Forecast the next day from at least 30 days of raw weather observations."""
    if not model_is_ready():
        raise FileNotFoundError(
            "Baseline deployment artifacts are missing. Run "
            "`uv run python experiments/train_baseline.py` first."
        )

    artifacts = joblib.load(ARTIFACTS_PATH)
    feature_columns = artifacts["feature_columns"]
    window = artifacts["window"]
    prepared = prepare_weather_dataframe(raw_weather)

    missing = sorted(set(feature_columns) - set(prepared.columns))
    if missing:
        raise ValueError(f"The uploaded CSV is missing required columns: {', '.join(missing)}")
    if len(prepared) < window:
        raise ValueError(f"At least {window} usable daily observations are required.")

    scaled = artifacts["feature_scaler"].transform(prepared[feature_columns])
    sequence = scaled[-window:].astype(np.float32)[np.newaxis, ...]
    model = tf.keras.models.load_model(MODEL_PATH)
    temp_scaled, rain_probability, rain_amount_scaled = model.predict(sequence, verbose=0)

    temperature = artifacts["temp_scaler"].inverse_transform(temp_scaled)[0, 0]
    log_rainfall = artifacts["amount_scaler"].inverse_transform(rain_amount_scaled)[0, 0]
    rainfall = max(0.0, float(np.expm1(log_rainfall)))
    probability = float(rain_probability[0, 0])

    return {
        "forecast_date": str(prepared.index[-1].date() + pd.Timedelta(days=1)),
        "temperature_c": float(temperature),
        "rain_probability": probability,
        "will_rain": probability >= 0.5,
        "rainfall_mm": rainfall,
    }
