import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.models.baseline_model import build_baseline_bilstm
from src.training.trainer import train_experiment


if __name__ == "__main__":
    train_experiment(
        model_builder=build_baseline_bilstm,
        experiment_name="baseline",
        use_weighted_loss=False,
    )
