"""
utils.py - shared helper functions. Every notebook imports this so that
all charts, tables, models and experiment logs are saved automatically
into Google Drive.
"""
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

# Project root inside Google Drive
ROOT = Path("/content/drive/MyDrive/network-intrusion-ml")

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
METRIC_DIR = ROOT / "results" / "metrics"
MODEL_DIR = ROOT / "models"
LOG_FILE = ROOT / "results" / "experiment_log.csv"

# Make sure every folder exists
for d in [RAW_DIR, PROCESSED_DIR, FIG_DIR, TABLE_DIR, METRIC_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42  # fixed seed so results are repeatable


def save_fig(name, dpi=150):
    """Save the current matplotlib figure to results/figures/<name>.png"""
    path = FIG_DIR / f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[saved figure] {path}")


def save_table(df, name, index=True):
    """Save a DataFrame to results/tables/<name>.csv"""
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"[saved table] {path}")


def log_experiment(experiment_id, task, model, features, preprocessing,
                   hyperparameters, split, metrics, notes=""):
    """Append one row to results/experiment_log.csv (the project document requires this)."""
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_id": experiment_id,
        "task": task,
        "model": model,
        "features": features,
        "preprocessing": preprocessing,
        "hyperparameters": str(hyperparameters),
        "split": split,
        "metrics": str(metrics),
        "notes": notes,
    }
    df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, mode="a", header=not LOG_FILE.exists(), index=False)
    print(f"[logged experiment] {experiment_id}")
