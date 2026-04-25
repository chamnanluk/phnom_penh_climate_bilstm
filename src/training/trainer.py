import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from config import (
    DATA_PATH,
    WINDOW,
    VAL_SIZE,
    TEST_SIZE,
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    RANDOM_SEED,
    RAIN_PROB_THRESHOLD,
    RAIN_WEIGHT_THRESHOLDS_MM,
    RAIN_WEIGHT_VALUES,
)
from src.data import load_and_prepare, build_dataset
from src.evaluate import evaluate_predictions
from src.utils.plotting import plot_series
from src.utils.seed import set_seed


def rainfall_mm_thresholds_to_scaled_log(amount_scaler, thresholds_mm):
    """Convert mm/day thresholds into scaled log-rainfall thresholds used by the model."""
    log_values = np.log1p(np.array(thresholds_mm, dtype=float)).reshape(-1, 1)
    return amount_scaler.transform(log_values).reshape(-1).tolist()


def train_experiment(model_builder, experiment_name: str, use_weighted_loss: bool = False):
    set_seed(RANDOM_SEED)

    output_dir = os.path.join("outputs", experiment_name)
    model_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    df = load_and_prepare(DATA_PATH)
    data = build_dataset(df, WINDOW, VAL_SIZE, TEST_SIZE)

    input_shape = (data["X_train"].shape[1], data["X_train"].shape[2])

    if use_weighted_loss:
        scaled_thresholds = rainfall_mm_thresholds_to_scaled_log(
            data["amount_scaler"], RAIN_WEIGHT_THRESHOLDS_MM
        )
        model = model_builder(
            input_shape=input_shape,
            learning_rate=LEARNING_RATE,
            scaled_thresholds=scaled_thresholds,
            threshold_weights=RAIN_WEIGHT_VALUES,
        )
    else:
        model = model_builder(input_shape=input_shape, learning_rate=LEARNING_RATE)

    print("\nModel summary:")
    model.summary()

    checkpoint_path = os.path.join(model_dir, f"best_{experiment_name}.keras")
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=5, factor=0.5),
        ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        data["X_train"],
        {
            "temp_output": data["y_temp_train"],
            "rain_status_output": data["y_status_train"],
            "rain_amount_output": data["y_amount_train"],
        },
        validation_data=(
            data["X_val"],
            {
                "temp_output": data["y_temp_val"],
                "rain_status_output": data["y_status_val"],
                "rain_amount_output": data["y_amount_val"],
            },
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    temp_pred, status_prob, amount_pred = model.predict(data["X_test"])

    results, pred_df = evaluate_predictions(
        data["y_temp_test"],
        data["y_status_test"],
        data["y_amount_test"],
        temp_pred,
        status_prob,
        amount_pred,
        data["temp_scaler"],
        data["amount_scaler"],
        threshold=RAIN_PROB_THRESHOLD,
    )

    pred_df.to_csv(os.path.join(output_dir, "predictions.csv"), index=False)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(output_dir, "feature_columns.json"), "w") as f:
        json.dump(data["feature_cols"], f, indent=2)

    with open(os.path.join(output_dir, "training_history.json"), "w") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    plot_series(
        pred_df["actual_temp"],
        pred_df["pred_temp"],
        f"{experiment_name}: Daily Average Temperature Forecast",
        "Temperature (°C)",
        os.path.join(output_dir, "temperature_forecast.png"),
    )
    plot_series(
        pred_df["actual_rain"],
        pred_df["pred_rain"],
        f"{experiment_name}: Daily Rainfall Forecast",
        "Rainfall (mm/day)",
        os.path.join(output_dir, "rainfall_forecast.png"),
    )

    print(f"\nFinal Test Metrics: {experiment_name}")
    for k, v in results.items():
        if v is None:
            print(f"{k}: None")
        else:
            print(f"{k}: {v:.4f}")

    return results
