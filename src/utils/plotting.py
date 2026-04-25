import matplotlib.pyplot as plt


def plot_series(actual, predicted, title, ylabel, out_path):
    plt.figure(figsize=(12, 5))
    plt.plot(actual, label="Actual")
    plt.plot(predicted, label="Predicted")
    plt.title(title)
    plt.xlabel("Test time step")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
