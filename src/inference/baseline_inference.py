import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.data import RAW_REQUIRED_COLUMNS, prepare_dataframe


class ArtifactError(RuntimeError):
    """Raised when model artifacts are missing or incompatible."""


class ValidationError(ValueError):
    """Raised when user input data fails validation."""


def validate_raw_input_dataframe(
    df: pd.DataFrame, window: int, required_columns=None
) -> pd.DataFrame:
    """Validate and sanitize raw model input dataframe."""
    required_columns = required_columns or RAW_REQUIRED_COLUMNS

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValidationError(
            f"Missing required columns: {', '.join(missing_cols)}"
        )

    cleaned = df[required_columns].copy()

    parsed_dates = pd.to_datetime(cleaned["Date"], errors="coerce")
    invalid_dates = int(parsed_dates.isna().sum())
    if invalid_dates:
        raise ValidationError(
            f"Found {invalid_dates} invalid date value(s) in 'Date' column."
        )
    cleaned["Date"] = parsed_dates

    numeric_cols = [c for c in required_columns if c != "Date"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    missing_numeric = {
        col: int(cleaned[col].isna().sum())
        for col in numeric_cols
        if cleaned[col].isna().any()
    }
    if missing_numeric:
        details = ", ".join(f"{k}: {v}" for k, v in missing_numeric.items())
        raise ValidationError(
            "Non-numeric or missing values found in required numeric columns "
            f"({details})."
        )

    if len(cleaned) < window:
        raise ValidationError(
            f"At least {window} rows are required, but only {len(cleaned)} row(s) were provided."
        )

    return cleaned.sort_values("Date")


@dataclass
class BaselineInferenceArtifacts:
    model_path: Path
    window: int
    feature_columns: list
    rain_prob_threshold: float
    required_input_columns: list
    feature_scaler: Any
    temp_scaler: Any
    amount_scaler: Any


class BaselineInferenceEngine:
    def __init__(
        self,
        model_path: str = "models/best_baseline.keras",
        scalers_path: str = "models/baseline_scalers.pkl",
        metadata_path: str = "models/baseline_metadata.json",
    ):
        self.artifacts = self._load_artifacts(
            model_path=Path(model_path),
            scalers_path=Path(scalers_path),
            metadata_path=Path(metadata_path),
        )
        self.model = self._load_model(self.artifacts.model_path)

    @staticmethod
    def _load_model(model_path: Path):
        try:
            import tensorflow as tf  # lazy import so tests can run without TensorFlow
        except ModuleNotFoundError as exc:
            raise ArtifactError(
                "TensorFlow is not installed. Install deployment dependencies before prediction."
            ) from exc

        if not model_path.exists():
            raise ArtifactError(
                f"Model file not found: {model_path}. Train/export baseline artifacts first."
            )

        try:
            return tf.keras.models.load_model(model_path)
        except Exception as exc:  # pragma: no cover
            raise ArtifactError(
                f"Failed to load model artifact at {model_path}: {exc}"
            ) from exc

    @staticmethod
    def _load_artifacts(
        model_path: Path, scalers_path: Path, metadata_path: Path
    ) -> BaselineInferenceArtifacts:
        if not metadata_path.exists():
            raise ArtifactError(
                f"Metadata file not found: {metadata_path}. Run baseline training/export first."
            )
        if not scalers_path.exists():
            raise ArtifactError(
                f"Scaler artifact not found: {scalers_path}. Run baseline training/export first."
            )

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        scalers = joblib.load(scalers_path)

        required_meta = ["window", "feature_columns", "rain_prob_threshold"]
        missing_meta = [k for k in required_meta if k not in metadata]
        if missing_meta:
            raise ArtifactError(
                "Metadata is incompatible. Missing keys: " + ", ".join(missing_meta)
            )

        required_scalers = ["feature_scaler", "temp_scaler", "amount_scaler"]
        missing_scalers = [k for k in required_scalers if k not in scalers]
        if missing_scalers:
            raise ArtifactError(
                "Scaler artifact is incompatible. Missing objects: "
                + ", ".join(missing_scalers)
            )

        metadata_model_path = metadata.get("model_path")
        resolved_model_path = Path(metadata_model_path) if metadata_model_path else model_path
        if not resolved_model_path.is_absolute():
            resolved_model_path = Path.cwd() / resolved_model_path

        return BaselineInferenceArtifacts(
            model_path=resolved_model_path,
            window=int(metadata["window"]),
            feature_columns=list(metadata["feature_columns"]),
            rain_prob_threshold=float(metadata["rain_prob_threshold"]),
            required_input_columns=list(
                metadata.get("required_input_columns", RAW_REQUIRED_COLUMNS)
            ),
            feature_scaler=scalers["feature_scaler"],
            temp_scaler=scalers["temp_scaler"],
            amount_scaler=scalers["amount_scaler"],
        )

    def predict_next_day(self, df: pd.DataFrame) -> Dict[str, Any]:
        validated = validate_raw_input_dataframe(
            df,
            window=self.artifacts.window,
            required_columns=self.artifacts.required_input_columns,
        )

        try:
            prepared = prepare_dataframe(validated)
        except Exception as exc:
            raise ValidationError(f"Failed to prepare input data: {exc}") from exc

        if len(prepared) < self.artifacts.window:
            raise ValidationError(
                "Not enough usable rows after preprocessing. "
                "Provide a longer date range with complete daily observations."
            )

        missing_features = [
            col for col in self.artifacts.feature_columns if col not in prepared.columns
        ]
        if missing_features:
            raise ArtifactError(
                "Prepared data is incompatible with model feature columns. Missing: "
                + ", ".join(missing_features[:5])
                + ("..." if len(missing_features) > 5 else "")
            )

        features = prepared[self.artifacts.feature_columns]
        scaled = self.artifacts.feature_scaler.transform(features)
        window_data = scaled[-self.artifacts.window :]
        X = window_data[np.newaxis, :, :].astype(np.float32)

        temp_scaled_pred, rain_prob_pred, amount_scaled_pred = self.model.predict(X, verbose=0)

        pred_temp = float(
            self.artifacts.temp_scaler.inverse_transform(temp_scaled_pred.reshape(-1, 1)).reshape(-1)[0]
        )
        rain_prob = float(rain_prob_pred.reshape(-1)[0])

        amount_log_pred = float(
            self.artifacts.amount_scaler.inverse_transform(amount_scaled_pred.reshape(-1, 1))
            .reshape(-1)[0]
        )
        rain_amount_raw = max(float(np.expm1(amount_log_pred)), 0.0)

        rain_status = int(rain_prob >= self.artifacts.rain_prob_threshold)
        final_rain_amount = rain_amount_raw if rain_status == 1 else 0.0

        last_date = prepared.index.max()
        next_date = (last_date + pd.Timedelta(days=1)).date().isoformat()

        return {
            "prediction_for_date": next_date,
            "temperature_c": pred_temp,
            "rain_probability": rain_prob,
            "rain_status": "Rain" if rain_status == 1 else "No Rain",
            "rainfall_amount_mm": final_rain_amount,
            "raw_rainfall_amount_mm": rain_amount_raw,
            "rain_threshold": self.artifacts.rain_prob_threshold,
            "window_size": self.artifacts.window,
        }
