"""Backward-compatible entry point: trains the baseline model."""
from experiments.train_baseline import train_experiment, build_baseline_bilstm


if __name__ == "__main__":
    train_experiment(
        model_builder=build_baseline_bilstm,
        experiment_name="baseline",
        use_weighted_loss=False,
    )
