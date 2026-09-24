import pandas as pd

from src.inference.baseline_inference import BaselineInferenceEngine


def predict_from_input_dataframe(df: pd.DataFrame) -> dict:
    engine = BaselineInferenceEngine()
    return engine.predict_next_day(df)
