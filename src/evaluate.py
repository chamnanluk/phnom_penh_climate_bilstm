import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_predictions(
    y_temp_test,
    y_status_test,
    y_amount_test,
    temp_scaled_pred,
    status_prob,
    amount_scaled_pred,
    temp_scaler,
    amount_scaler,
    threshold=0.5,
):
    actual_temp = temp_scaler.inverse_transform(y_temp_test.reshape(-1, 1)).reshape(-1)
    pred_temp = temp_scaler.inverse_transform(temp_scaled_pred).reshape(-1)

    status_pred = (status_prob.reshape(-1) >= threshold).astype(int)
    actual_status = y_status_test.astype(int)

    amount_log_pred = amount_scaler.inverse_transform(amount_scaled_pred).reshape(-1)
    amount_pred = np.expm1(amount_log_pred)
    amount_pred = np.maximum(amount_pred, 0)

    # Operational final rainfall forecast: no rain status means amount = 0.
    final_rain_pred = np.where(status_pred == 1, amount_pred, 0)

    actual_amount_log = amount_scaler.inverse_transform(y_amount_test.reshape(-1, 1)).reshape(-1)
    actual_rain = np.expm1(actual_amount_log)

    rainy_mask = actual_status == 1
    heavy_mask = actual_rain >= 20

    results = {
        "temperature_mae": float(mean_absolute_error(actual_temp, pred_temp)),
        "temperature_rmse": rmse(actual_temp, pred_temp),
        "rainfall_mae": float(mean_absolute_error(actual_rain, final_rain_pred)),
        "rainfall_rmse": rmse(actual_rain, final_rain_pred),
        "rainfall_mae_rainy_days": float(mean_absolute_error(actual_rain[rainy_mask], final_rain_pred[rainy_mask])) if rainy_mask.any() else None,
        "rainfall_rmse_rainy_days": rmse(actual_rain[rainy_mask], final_rain_pred[rainy_mask]) if rainy_mask.any() else None,
        "rainfall_mae_heavy_days_20mm": float(mean_absolute_error(actual_rain[heavy_mask], final_rain_pred[heavy_mask])) if heavy_mask.any() else None,
        "rainfall_rmse_heavy_days_20mm": rmse(actual_rain[heavy_mask], final_rain_pred[heavy_mask]) if heavy_mask.any() else None,
        "rain_status_accuracy": float(accuracy_score(actual_status, status_pred)),
        "rain_status_precision": float(precision_score(actual_status, status_pred, zero_division=0)),
        "rain_status_recall": float(recall_score(actual_status, status_pred, zero_division=0)),
        "rain_status_f1": float(f1_score(actual_status, status_pred, zero_division=0)),
    }

    pred_df = pd.DataFrame({
        "actual_temp": actual_temp,
        "pred_temp": pred_temp,
        "actual_rain": actual_rain,
        "pred_rain": final_rain_pred,
        "raw_pred_rain_amount": amount_pred,
        "actual_rain_status": actual_status,
        "pred_rain_status": status_pred,
        "rain_probability": status_prob.reshape(-1),
    })
    return results, pred_df
