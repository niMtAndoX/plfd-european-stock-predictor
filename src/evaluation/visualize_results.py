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
# 1. BAR PLOT — Model Ranking by RMSE / MSE / MAE
# ---------------------------------------------------------------------------

def _plot_metric_bar(df_results: pd.DataFrame, metric: str, out_path=None, title_suffix: str = ""):
    if metric not in df_results.columns:
        print(f"[visualize_results] Metric '{metric}' not in leaderboard, skipping bar plot.")
        return

    df = df_results.sort_values(metric)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x="model", y=metric, data=df, palette="viridis")

    title = f"Model Ranking by {metric.upper()}"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Model")
    ax.set_ylabel(metric.upper())
    plt.xticks(rotation=45, ha="right")

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    plt.close()  # avoid interactive popping when running in batch


def plot_rmse_bar(df_results: pd.DataFrame, out_path=None):
    _plot_metric_bar(df_results, "rmse", out_path)


def plot_mse_bar(df_results: pd.DataFrame, out_path=None):
    _plot_metric_bar(df_results, "mse", out_path)


def plot_mae_bar(df_results: pd.DataFrame, out_path=None):
    _plot_metric_bar(df_results, "mae", out_path)


# ---------------------------------------------------------------------------
# 2. HEATMAP — Multi-Metric Comparison (RMSE, MAE, MSE)
# ---------------------------------------------------------------------------

def plot_metric_heatmap(df_results: pd.DataFrame, out_path=None):
    """Professional-looking heatmap over available metrics."""
    # choose subset of metrics actually present
    candidate_metrics = ["rmse", "mae", "mse"]
    metrics = [m for m in candidate_metrics if m in df_results.columns]
    if not metrics:
        print("[visualize_results] No suitable metrics for heatmap, skipping.")
        return

    df = df_results.set_index("model")[metrics]

    plt.figure(figsize=(10, 0.6 * len(df.index) + 3))
    sns.heatmap(
        df,
        annot=True,
        cmap="mako_r",
        fmt=".4f",
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Error"},
        square=False,
    )

    plt.title("Model Performance Heatmap", fontsize=16, weight="bold", pad=12)
    plt.ylabel("Model")
    plt.xlabel("Metric")

    if out_path:
        plt.tight_layout()
        plt.savefig(out_path)
    plt.close()


# ---------------------------------------------------------------------------
# 3. CLASS-SPECIFIC BAR CHARTS (per model class, by RMSE)
# ---------------------------------------------------------------------------

# Map individual models to high-level classes
MODEL_CLASS_MAP = {
    # decision trees / tree ensembles
    "RANDOM_FOREST": "decision_tree",
    "XGBOOST": "decision_tree",
    # CNN-like
    "CNN": "cnn",
    "SACLSTM": "cnn",
    "SCINET": "cnn",
    # RNN-like
    "RNN": "rnn",
    "MTSMFF": "rnn",
    "DILATED_RNN": "rnn",
    # Transformers
    "TRANSFORMER": "transformer",
    "TFT": "transformer",
    "PYRAFORMER": "transformer",
    "PREFORMER": "transformer",
    "AUTOFORMER": "transformer",
}


def plot_class_rmse_bars(df_results: pd.DataFrame, out_dir: Path):
    """Create RMSE bar charts per model class (decision_tree, cnn, rnn, transformer)."""
    if "rmse" not in df_results.columns:
        print("[visualize_results] rmse not in leaderboard, skipping class-specific bars.")
        return

    df = df_results.copy()
    df["model_class"] = df["model"].map(MODEL_CLASS_MAP)

    for cls in ["decision_tree", "cnn", "rnn", "transformer"]:
        sub = df[df["model_class"] == cls]
        if sub.empty:
            continue

        out_path = out_dir / f"rmse_bar_{cls}.png"
        _plot_metric_bar(sub, "rmse", out_path=out_path, title_suffix=f"{cls.title()} Models")


# ---------------------------------------------------------------------------
# 4. TABLES — Metric tables saved as PNG (RMSE, MSE, MAE)
# ---------------------------------------------------------------------------

def save_metric_table(df_results: pd.DataFrame, metric: str, out_path: Path):
    """Render a simple, clean table (model vs metric) and save as PNG."""
    if metric not in df_results.columns:
        print(f"[visualize_results] Metric '{metric}' not in leaderboard, skipping table.")
        return

    df = df_results[["model", metric]].sort_values(metric).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 0.4 * len(df) + 1.5))
    ax.axis("tight")
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns.str.upper(),
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)

    # Bold header row
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")

    ax.set_title(f"{metric.upper()} by Model", fontsize=14, weight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. (Existing) Additional plots left unchanged (boxplot, residuals, etc.)
# ---------------------------------------------------------------------------

# ...existing definitions: plot_cv_boxplot, plot_residual_kde,
# plot_predictions, plot_scatter, plot_cumulative_error...


def main():
    """
    Load leaderboard results and write a set of standard plots to disk.
    All plots are saved as PNG files in: <inner-project-root>/results/
    Expects a CSV at: <inner-project-root>/results/leaderboard.csv
    """
    # src_dir: .../plfd-european-stock-predictor/plfd-european-stock-predictor/src
    src_dir = Path(__file__).resolve().parents[1]
    # Inner project root: .../plfd-european-stock-predictor/plfd-european-stock-predictor
    project_root = src_dir.parent
    results_dir = project_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    leaderboard_csv = results_dir / "leaderboard.csv"
    if not leaderboard_csv.exists():
        print(f"[visualize_results] Leaderboard CSV not found: {leaderboard_csv}")
        return

    df_results = pd.read_csv(leaderboard_csv)

    # Overall bar charts
    plot_rmse_bar(df_results, out_path=results_dir / "rmse_bar.png")
    plot_mse_bar(df_results, out_path=results_dir / "mse_bar.png")
    plot_mae_bar(df_results, out_path=results_dir / "mae_bar.png")

    # Nicer heatmap (no radar chart anymore)
    plot_metric_heatmap(df_results, out_path=results_dir / "metric_heatmap.png")

    # Class-specific RMSE bar charts
    plot_class_rmse_bars(df_results, results_dir)

    # Metric tables
    save_metric_table(df_results, "rmse", results_dir / "rmse_table.png")
    save_metric_table(df_results, "mse", results_dir / "mse_table.png")
    save_metric_table(df_results, "mae", results_dir / "mae_table.png")

    print(f"[visualize_results] Figures and tables written to {results_dir}")


if __name__ == "__main__":
    main()
