import os
import json
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)

EXPERIMENTS = {
    "Baseline BiLSTM": "outputs/baseline/metrics.json",
    "BiLSTM + Attention + Weighted Loss": "outputs/attention_weighted/metrics.json",
}

ORDERED_COLUMNS = [
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
]


def main():
    rows = []
    for model_name, path in EXPERIMENTS.items():
        if not os.path.exists(path):
            print(f"Missing {path}. Run the training script first.")
            continue
        with open(path, "r") as f:
            metrics = json.load(f)
        metrics["model"] = model_name
        rows.append(metrics)

    if not rows:
        raise SystemExit("No metrics found. Run train_baseline.py and train_attention_weighted.py first.")

    comparison = pd.DataFrame(rows)
    cols = ["model"] + [c for c in ORDERED_COLUMNS if c in comparison.columns]
    comparison = comparison[cols]

    os.makedirs("outputs/comparison", exist_ok=True)
    comparison.to_csv("outputs/comparison/model_comparison.csv", index=False)

    print("\nModel comparison")
    print(comparison.to_string(index=False))

    # Simple comparison chart for core metrics.
    core = comparison.set_index("model")[[
        "temperature_mae", "rainfall_mae", "rainfall_rmse", "rain_status_f1"
    ]]
    ax = core.plot(kind="bar", figsize=(12, 5))
    ax.set_title("Model Comparison: Core Metrics")
    ax.set_ylabel("Metric value")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/comparison/model_comparison_core_metrics.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
