"""
Train Hybrid SARIMA + BiLSTM model for daily temperature forecasting.

Workflow:
1. Load NASA weather data.
2. Fit SARIMA on training temperature.
3. Compute SARIMA residuals.
4. Train BiLSTM to predict residuals using multivariate weather sequences.
5. Final forecast = SARIMA forecast + BiLSTM residual forecast.
6. Save metrics, predictions, and plots.

Run:
    python experiments/train_hybrid_sarima_bilstm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Allow running from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.hybrid.hybrid_data import (
    prepare_hybrid_dataframe,
    create_residual_sequences,
    scale_features,
)
from src.hybrid.sarima_utils import fit_sarima, sarima_predict, save_sarima_summary
from src.hybrid.residual_bilstm import ResidualBiLSTM


# -----------------------
# Config
# -----------------------
DATA_PATH = PROJECT_ROOT / "data" / "Dataset_with_RainStatus_V5.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hybrid_sarima_bilstm"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "T2M"          # If missing, created from T2M_MAX/T2M_MIN or T2M_AVG
WINDOW = 30
TRAIN_RATIO = 0.85

# Daily annual seasonality can be slow. If training is too slow, use (1,0,1,7).
SARIMA_ORDER = (2, 0, 2)
SARIMA_SEASONAL_ORDER = (1, 0, 1, 7)

EPOCHS = 80
BATCH_SIZE = 64
LR = 0.001
PATIENCE = 10
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def get_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    exclude = {
        target_col,
        "RainStatus",
        "rain_status",
        "rain_status_target",
        "rain_amount_log",
    }
    # Keep numeric columns only
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    return feature_cols


def train_residual_model(model, X_train, y_train, X_val, y_val, device):
    train_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
    )

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.HuberLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    best_state = None
    patience_count = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def main():
    print("Loading data...")
    df = prepare_hybrid_dataframe(DATA_PATH, target_col=TARGET_COL)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)

    train_df = df.iloc[:train_end].copy()
    test_df = df.iloc[train_end:].copy()

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")

    # -----------------------
    # 1. SARIMA baseline
    # -----------------------
    print("\nFitting SARIMA...")
    sarima_fit = fit_sarima(
        train_series=train_df[TARGET_COL],
        order=SARIMA_ORDER,
        seasonal_order=SARIMA_SEASONAL_ORDER,
    )

    sarima_config = {
        "target_col": TARGET_COL,
        "order": SARIMA_ORDER,
        "seasonal_order": SARIMA_SEASONAL_ORDER,
        "window": WINDOW,
        "train_ratio": TRAIN_RATIO,
    }
    save_sarima_summary(sarima_fit, OUTPUT_DIR, sarima_config)

    # In-sample SARIMA predictions for train
    train_sarima_pred = sarima_predict(
        sarima_fit,
        start=0,
        end=len(train_df) - 1,
    )

    # Out-of-sample SARIMA forecast for test
    test_sarima_pred = np.asarray(
        sarima_fit.get_forecast(steps=len(test_df)).predicted_mean,
        dtype=float,
    )

    train_actual = train_df[TARGET_COL].values.astype(float)
    test_actual = test_df[TARGET_COL].values.astype(float)

    train_residuals = train_actual - train_sarima_pred
    test_residuals = test_actual - test_sarima_pred

    # -----------------------
    # 2. Residual BiLSTM
    # -----------------------
    feature_cols = get_feature_columns(df, TARGET_COL)
    X_train_scaled, X_test_scaled, feature_scaler = scale_features(train_df, test_df, feature_cols)

    X_train_seq, y_train_resid, train_sarima_seq, train_actual_seq = create_residual_sequences(
        X_train_scaled,
        train_residuals,
        train_sarima_pred,
        train_actual,
        window=WINDOW,
    )

    X_test_seq, y_test_resid, test_sarima_seq, test_actual_seq = create_residual_sequences(
        X_test_scaled,
        test_residuals,
        test_sarima_pred,
        test_actual,
        window=WINDOW,
    )

    # Train/validation split inside residual sequences
    val_start = int(len(X_train_seq) * 0.85)

    X_res_train = X_train_seq[:val_start]
    y_res_train = y_train_resid[:val_start]
    X_res_val = X_train_seq[val_start:]
    y_res_val = y_train_resid[val_start:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining residual BiLSTM on device: {device}")

    model = ResidualBiLSTM(input_size=X_train_seq.shape[2]).to(device)

    history = train_residual_model(
        model,
        X_res_train,
        y_res_train,
        X_res_val,
        y_res_val,
        device,
    )

    torch.save(model.state_dict(), MODEL_DIR / "hybrid_residual_bilstm.pth")

    # -----------------------
    # 3. Prediction
    # -----------------------
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test_seq, dtype=torch.float32).to(device)
        residual_pred = model(X_test_tensor).cpu().numpy().reshape(-1)

    sarima_component = test_sarima_seq.reshape(-1)
    actual_temp = test_actual_seq.reshape(-1)
    hybrid_pred = sarima_component + residual_pred

    sarima_only_pred = sarima_component

    # -----------------------
    # 4. Evaluation
    # -----------------------
    sarima_mae = mean_absolute_error(actual_temp, sarima_only_pred)
    sarima_rmse = np.sqrt(mean_squared_error(actual_temp, sarima_only_pred))

    hybrid_mae = mean_absolute_error(actual_temp, hybrid_pred)
    hybrid_rmse = np.sqrt(mean_squared_error(actual_temp, hybrid_pred))

    residual_mae = mean_absolute_error(y_test_resid.reshape(-1), residual_pred)
    residual_rmse = np.sqrt(mean_squared_error(y_test_resid.reshape(-1), residual_pred))

    metrics = {
        "model": "Hybrid SARIMA + BiLSTM",
        "temperature_mae": float(hybrid_mae),
        "temperature_rmse": float(hybrid_rmse),
        "sarima_only_temperature_mae": float(sarima_mae),
        "sarima_only_temperature_rmse": float(sarima_rmse),
        "residual_mae": float(residual_mae),
        "residual_rmse": float(residual_rmse),
        "sarima_order": str(SARIMA_ORDER),
        "sarima_seasonal_order": str(SARIMA_SEASONAL_ORDER),
        "window": WINDOW,
    }

    print("\nFinal Hybrid Test Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    pred_df = pd.DataFrame({
        "date": test_df.index[WINDOW:],
        "actual_temperature": actual_temp,
        "sarima_prediction": sarima_only_pred,
        "residual_prediction": residual_pred,
        "hybrid_prediction": hybrid_pred,
        "actual_residual": y_test_resid.reshape(-1),
    })
    pred_df.to_csv(OUTPUT_DIR / "predictions.csv", index=False)

    # -----------------------
    # 5. Plots
    # -----------------------
    plt.figure(figsize=(12, 5))
    plt.plot(pred_df["date"], pred_df["actual_temperature"], label="Actual Temperature")
    plt.plot(pred_df["date"], pred_df["sarima_prediction"], label="SARIMA")
    plt.plot(pred_df["date"], pred_df["hybrid_prediction"], label="SARIMA + BiLSTM")
    plt.title("Hybrid SARIMA + BiLSTM Temperature Forecast")
    plt.xlabel("Date")
    plt.ylabel("Temperature")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "hybrid_temperature_forecast.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    labels = ["SARIMA", "Hybrid SARIMA + BiLSTM"]
    mae_values = [sarima_mae, hybrid_mae]
    rmse_values = [sarima_rmse, hybrid_rmse]

    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width / 2, mae_values, width, label="MAE")
    plt.bar(x + width / 2, rmse_values, width, label="RMSE")
    plt.xticks(x, labels)
    plt.ylabel("Temperature Error")
    plt.title("SARIMA vs Hybrid Temperature Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sarima_vs_hybrid_metrics.png", dpi=300)
    plt.close()

    print(f"\nSaved hybrid outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
