import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data import RAW_REQUIRED_COLUMNS, prepare_dataframe
from src.inference.baseline_inference import (
    ArtifactError,
    BaselineInferenceEngine,
    ValidationError,
    validate_raw_input_dataframe,
)
from src.inference.app_service import predict_from_input_dataframe


class FakeModel:
    def predict(self, X, verbose=0):
        batch = X.shape[0]
        return (
            np.full((batch, 1), 0.5, dtype=np.float32),
            np.full((batch, 1), 0.8, dtype=np.float32),
            np.full((batch, 1), 0.4, dtype=np.float32),
        )


def make_raw_dataframe(rows=120):
    start = pd.Timestamp("2020-01-01")
    dates = pd.date_range(start=start, periods=rows, freq="D")
    data = {
        "Date": dates,
        "PRECTOTCORR": np.linspace(0, 20, rows),
        "WS2M": np.linspace(1, 5, rows),
        "T2M_RANGE": np.linspace(5, 12, rows),
        "T2M_MAX": np.linspace(29, 35, rows),
        "T2M_MIN": np.linspace(22, 26, rows),
        "PS": np.linspace(100, 102, rows),
        "ALLSKY_SFC_SW_DWN": np.linspace(12, 18, rows),
        "RH2M": np.linspace(60, 85, rows),
        "T2MDEW": np.linspace(19, 24, rows),
    }
    return pd.DataFrame(data)


class InferenceTests(unittest.TestCase):
    def test_validate_input_missing_column(self):
        df = make_raw_dataframe(40).drop(columns=["WS2M"])
        with self.assertRaises(ValidationError):
            validate_raw_input_dataframe(df, window=30)

    def test_missing_artifacts_raise_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ArtifactError):
                BaselineInferenceEngine(
                    model_path=f"{tmpdir}/best_baseline.keras",
                    scalers_path=f"{tmpdir}/baseline_scalers.pkl",
                    metadata_path=f"{tmpdir}/baseline_metadata.json",
                )

    def test_prediction_path_with_mock_model(self):
        raw = make_raw_dataframe(120)
        prepared = prepare_dataframe(raw)
        feature_cols = [
            c
            for c in prepared.columns
            if c not in ["PRECTOTCORR", "RainStatus", "rain_status_target", "rain_amount_log"]
        ]

        feature_scaler = MinMaxScaler().fit(prepared[feature_cols])
        temp_scaler = MinMaxScaler().fit(prepared[["T2M_AVG"]])
        amount_scaler = MinMaxScaler().fit(prepared[["rain_amount_log"]])

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            model_path = tmp / "best_baseline.keras"
            model_path.write_text("placeholder")

            scalers_path = tmp / "baseline_scalers.pkl"
            joblib.dump(
                {
                    "feature_scaler": feature_scaler,
                    "temp_scaler": temp_scaler,
                    "amount_scaler": amount_scaler,
                },
                scalers_path,
            )

            metadata_path = tmp / "baseline_metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_path": str(model_path),
                        "window": 30,
                        "feature_columns": feature_cols,
                        "rain_prob_threshold": 0.5,
                        "required_input_columns": RAW_REQUIRED_COLUMNS,
                    }
                )
            )

            with patch.object(BaselineInferenceEngine, "_load_model", return_value=FakeModel()):
                engine = BaselineInferenceEngine(
                    model_path=str(model_path),
                    scalers_path=str(scalers_path),
                    metadata_path=str(metadata_path),
                )
                result = engine.predict_next_day(raw)

            self.assertIn("temperature_c", result)
            self.assertEqual(result["rain_status"], "Rain")
            self.assertGreaterEqual(result["rainfall_amount_mm"], 0.0)

    def test_app_facing_prediction_smoke(self):
        raw = make_raw_dataframe(40)

        class DummyEngine:
            def predict_next_day(self, _df):
                return {
                    "prediction_for_date": "2020-02-10",
                    "temperature_c": 30.0,
                    "rain_probability": 0.3,
                    "rain_status": "No Rain",
                    "rainfall_amount_mm": 0.0,
                }

        with patch("src.inference.app_service.BaselineInferenceEngine", return_value=DummyEngine()):
            result = predict_from_input_dataframe(raw)

        self.assertEqual(result["rain_status"], "No Rain")


if __name__ == "__main__":
    unittest.main()
