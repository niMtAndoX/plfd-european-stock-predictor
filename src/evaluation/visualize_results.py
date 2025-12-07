"""
Advanced visualization suite for model comparison.
Publication-quality figures for research papers.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12


# ---------------------------------------------------------------------------
# 1. BAR PLOT — Model Ranking by RMSE
# ---------------------------------------------------------------------------

def plot_rmse_bar(df_results: pd.DataFrame, out_path=None):
    df = df_results.sort_values("rmse")

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="model", y="rmse", data=df, palette="viridis")

    ax.set_title("Model Ranking by RMSE", fontsize=16, weight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel("RMSE")
    plt.xticks(rotation=45, ha="right")

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    plt.show()


# ---------------------------------------------------------------------------
# 2. HEATMAP — Multi-Metric Comparison (RMSE, MAE, MAPE, R2)
# ---------------------------------------------------------------------------

def plot_metric_heatmap(df_results: pd.DataFrame, out_path=None):
    metrics = ["rmse", "mae", "mape", "r2"]
    df = df_results.set_index("model")[metrics]

    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, cmap="coolwarm", fmt=".4f", linewidths=0.5)

    plt.title("Model Performance Heatmap")
    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    plt.show()


# ---------------------------------------------------------------------------
# 3. RADAR CHART — Multi-Metric Performance Trade-Off
# ---------------------------------------------------------------------------

def radar_chart(df_results: pd.DataFrame, out_path=None):
    from math import pi

    df = df_results.copy()
    metrics = ["rmse", "mae", "mape", "r2"]

    df_scaled = df[metrics].copy()

    # scale metrics 0–1 so radar chart is meaningful
    for col in metrics:
        df_scaled[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    categories = metrics
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)] + [0]

    plt.figure(figsize=(9, 9))
    ax = plt.subplot(111, polar=True)

    for _, row in df_scaled.iterrows():
        vals = row.values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, label=row["model"])
        ax.fill(angles, vals, alpha=0.1)

    plt.title("Radar Chart: Model Metric Comparison")
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 4. BOXPLOT — Cross-Validation RMSE Distribution
# ---------------------------------------------------------------------------

def plot_cv_boxplot(cv_results_path: str | Path, out_path=None):
    """
    cv_results_path: JSON storing RMSE per fold per model.
    Format example:
    {
       "CNN": {"folds": [0.12, 0.15, 0.14, 0.13, 0.14]},
       ...
    }
    """
    import json

    with open(cv_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for model_name, info in data.items():
        for rmse in info["folds"]:
            records.append({"model": model_name, "rmse": rmse})

    df = pd.DataFrame(records)

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="model", y="rmse", data=df, palette="Set3")
    sns.swarmplot(x="model", y="rmse", data=df, color="black", alpha=0.6)

    plt.title("Cross-Validation RMSE Distribution")
    plt.xticks(rotation=45, ha="right")
    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    plt.show()


# ---------------------------------------------------------------------------
# 5. KDE — Residual Distribution (Error Curve)
# ---------------------------------------------------------------------------

def plot_residual_kde(y_true, y_pred, model_name, out_path=None):
    residuals = y_true - y_pred

    plt.figure(figsize=(8, 5))
    sns.kdeplot(residuals, fill=True, color="blue", alpha=0.6, linewidth=2)

    plt.title(f"Residual Density — {model_name}")
    plt.xlabel("Prediction Error (Residual)")
    plt.ylabel("Density")

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    plt.show()


# ---------------------------------------------------------------------------
# 6. TIME-SERIES PLOT — Ground Truth vs Predictions (top 3 models)
# ---------------------------------------------------------------------------

def plot_predictions(df_test, predictions: dict, out_path=None):
    """
    predictions = {model_name: y_pred_array}
    """
    plt.figure(figsize=(14, 6))

    plt.plot(df_test["Date"], df_test["Return_t"], label="Actual", color="black", linewidth=2)

    for model_name, pred in predictions.items():
        plt.plot(df_test["Date"], pred, label=model_name, linewidth=1)

    plt.title("Ground Truth vs Predicted Returns")
    plt.legend()
    plt.xlabel("Date")
    plt.ylabel("Return")

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    plt.show()


# ---------------------------------------------------------------------------
# 7. SCATTER PLOT — Predicted vs Actual
# ---------------------------------------------------------------------------

def plot_scatter(y_true, y_pred, model_name, out_path=None):
    plt.figure(figsize=(6, 6))

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    sns.lineplot(x=y_true, y=y_true, color="red", label="Ideal")

    plt.title(f"Predicted vs Actual — {model_name}")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    plt.show()


# ---------------------------------------------------------------------------
# 8. CUMULATIVE ERROR CURVE
# ---------------------------------------------------------------------------

def plot_cumulative_error(y_true, y_pred, model_name, out_path=None):
    cumulative_error = np.cumsum(np.abs(y_true - y_pred))

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_error, label=model_name, color="blue")

    plt.title(f"Cumulative Absolute Error — {model_name}")
    plt.xlabel("Time")
    plt.ylabel("Cum. Error")

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    plt.show()
