"""
Compare Baseline BiLSTM, Attention + Weighted Loss, and Hybrid SARIMA + BiLSTM.

Run after:
    python experiments/train_baseline.py
    python experiments/train_attention_weighted.py
    python experiments/train_hybrid_sarima_bilstm.py

Then:
    python experiments/compare_all_models.py
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
COMPARISON_DIR = OUTPUT_DIR / "comparison"
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


def load_metrics(path: Path) -> dict | None:
    if not path.exists():
        print(f"Missing: {path}")
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    metric_files = [
        ("Baseline BiLSTM", OUTPUT_DIR / "baseline" / "metrics.json"),
        ("BiLSTM + Attention + Weighted Loss", OUTPUT_DIR / "attention_weighted" / "metrics.json"),
        ("Hybrid SARIMA + BiLSTM", OUTPUT_DIR / "hybrid_sarima_bilstm" / "metrics.json"),
    ]

    rows = []
    for name, path in metric_files:
        metrics = load_metrics(path)
        if metrics is None:
            continue
        metrics["model"] = name
        rows.append(metrics)

    if not rows:
        raise FileNotFoundError("No metrics files found. Train models first.")

    df = pd.DataFrame(rows)

    # Put common columns first where available
    common_cols = [
        "model",
        "temperature_mae",
        "temperature_rmse",
        "rainfall_mae",
        "rainfall_rmse",
        "rainfall_mae_rainy_days",
        "rainfall_rmse_rainy_days",
        "rainfall_mae_heavy_days_20mm",
        "rainfall_rmse_heavy_days_20mm",
        "rain_status_accuracy",
        "rain_status_precision",
        "rain_status_recall",
        "rain_status_f1",
        "sarima_only_temperature_mae",
        "sarima_only_temperature_rmse",
        "residual_mae",
        "residual_rmse",
    ]

    cols = [c for c in common_cols if c in df.columns] + [c for c in df.columns if c not in common_cols]
    df = df[cols]

    output_csv = COMPARISON_DIR / "all_model_comparison.csv"
    df.to_csv(output_csv, index=False)

    print("\nAll Model Comparison")
    print(df.to_string(index=False))
    print(f"\nSaved: {output_csv}")

    # Plot temperature-only comparison, because Hybrid model is temperature-focused.
    if {"temperature_mae", "temperature_rmse"}.issubset(df.columns):
        plot_df = df.dropna(subset=["temperature_mae", "temperature_rmse"]).copy()
        x = np.arange(len(plot_df))
        width = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(x - width / 2, plot_df["temperature_mae"], width, label="MAE")
        plt.bar(x + width / 2, plot_df["temperature_rmse"], width, label="RMSE")
        plt.xticks(x, plot_df["model"], rotation=20, ha="right")
        plt.ylabel("Temperature Error")
        plt.title("Temperature Forecasting Comparison Across Models")
        plt.legend()
        plt.tight_layout()
        plt.savefig(COMPARISON_DIR / "temperature_model_comparison.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()
