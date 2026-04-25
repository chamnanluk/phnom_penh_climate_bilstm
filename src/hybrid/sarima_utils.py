"""
SARIMA utilities for Hybrid SARIMA + BiLSTM residual learning.

This module:
1. fits SARIMA/SARIMAX on the temperature target,
2. generates in-sample and test SARIMA predictions,
3. computes residuals = actual - SARIMA prediction,
4. saves SARIMA model artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import json
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarima(
    train_series: pd.Series,
    order: Tuple[int, int, int] = (2, 0, 2),
    seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, 365),
    enforce_stationarity: bool = False,
    enforce_invertibility: bool = False,
):
    """
    Fit SARIMA model to a training temperature series.

    For daily weather data, seasonality can be annual:
        seasonal_order=(P, D, Q, 365)

    If the dataset is small or fitting is slow, change seasonal period to 7
    for weekly seasonality:
        seasonal_order=(1, 0, 1, 7)
    """
    train_series = train_series.astype(float).dropna()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=enforce_stationarity,
            enforce_invertibility=enforce_invertibility,
        )
        fitted = model.fit(disp=False)

    return fitted


def sarima_predict(
    fitted_model,
    start: int,
    end: int,
) -> np.ndarray:
    """Return SARIMA prediction as numpy array."""
    pred = fitted_model.predict(start=start, end=end)
    return np.asarray(pred, dtype=float)


def save_sarima_summary(fitted_model, output_dir: str | Path, config: Dict[str, Any]) -> None:
    """Save SARIMA summary and configuration."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "sarima_summary.txt").write_text(str(fitted_model.summary()), encoding="utf-8")
    (output_dir / "sarima_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )
