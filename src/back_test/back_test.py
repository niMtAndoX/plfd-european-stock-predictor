from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import sys
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = SRC_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_selection.model_registry import MODEL_REGISTRY

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
BACKTEST_MAX_EPOCHS = int(os.getenv("BACKTEST_MAX_EPOCHS", "20"))

TOP_MODELS = ["XGBOOST", "RANDOM_FOREST", "DILATED_RNN", "SACLSTM", "MTSMFF"]

TRANSACTION_COST = 0.001      # 10 bps per position change
TARGET_DAILY_VOL = 0.01       # 1% volatility targeting
TRADING_DAYS = 252.0
INITIAL_CAPITAL = 10_000.0    # starting capital for threshold backtest

# Asian feature columns we want to use as model inputs
ASIAN_FEATURE_COLS = [
    "Return_t_1",
    "Return_t_2",
    "Return_t_3",
    "SMA_5",
    "SMA_10",
    "STD_5",
]


def load_stoxx600_with_asian_features(data_dir: Path) -> pd.DataFrame:
    """
    Build a DataFrame with:
      - target: STOXX600 Return_t
      - features: aggregated Asian features listed in ASIAN_FEATURE_COLS
      - Date column preserved for chronological splits

    Data is loaded from the combined CSV produced by the ETL step:
      <inner-project-root>/src/data_processing/data/clean_features/STOXX600_with_asian_features.csv
    """
    combined_path = data_dir / "STOXX600_with_asian_features.csv"

    if not combined_path.exists():
        raise FileNotFoundError(f"[back_test] Combined features CSV not found: {combined_path}")

    df = pd.read_csv(combined_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Ensure expected feature columns exist
    missing = set(ASIAN_FEATURE_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"[back_test] Missing expected feature columns: {missing}")

    # Keep only Date, target, and Asian features (others are safe to keep as-is if present)
    df = df[["Date", "Return_t"] + ASIAN_FEATURE_COLS].dropna().reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_data_and_results() -> Tuple[pd.DataFrame, Dict]:
    """
    Returns:
      df: merged STOXX target + Asian (aggregated) features only
      selection_results: JSON with best params per model
    """
    clean_dir = SRC_DIR / "data_processing" / "data" / "clean_features"
    results_json = PROJECT_ROOT / "results" / "model_selection_results.json"

    if not results_json.exists():
        raise FileNotFoundError(results_json)

    # >>> unified loader: STOXX target + Asian (SSE) features only <<<
    df = load_stoxx600_with_asian_features(clean_dir)

    with results_json.open("r", encoding="utf-8") as f:
        selection_results = json.load(f)

    return df, selection_results


def annualized_return(equity: np.ndarray) -> float:
    years = len(equity) / TRADING_DAYS
    if years <= 0 or equity[-1] <= 0:
        return np.nan
    return equity[-1] ** (1.0 / years) - 1.0


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    return drawdown.min()


# ---------------------------------------------------------------------
# Robust trading strategy (volatility-scaled, starts at 1.0)
# ---------------------------------------------------------------------
def backtest_strategy(
    returns: np.ndarray,
    preds: np.ndarray,
    pos_th: float,
    neg_th: float,
    cost: float = TRANSACTION_COST,
    target_vol: float = TARGET_DAILY_VOL,
) -> np.ndarray:
    """
    Volatility-scaled long/cash strategy with transaction costs.
    """
    pos = 0.0
    prev_pos = 0.0
    equity = [1.0]

    rolling_vol = pd.Series(returns).rolling(20).std().values

    for r, p, vol in zip(returns, preds, rolling_vol):
        if np.isnan(vol) or vol == 0:
            vol = target_vol

        # signal → desired position
        if p > pos_th:
            desired_pos = min(abs(p) / vol, 1.0)
        elif p < neg_th:
            desired_pos = 0.0
        else:
            desired_pos = pos

        turnover = abs(desired_pos - prev_pos)
        net_ret = desired_pos * r - turnover * cost

        equity.append(equity[-1] * (1.0 + net_ret))

        prev_pos = desired_pos
        pos = desired_pos

    return np.asarray(equity[1:])


# ---------------------------------------------------------------------
# Simple threshold strategy (long / cash, starts at 10k EUR)
# ---------------------------------------------------------------------
def backtest_threshold_strategy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    neg_th: float = -0.001,      # -0.10%
    pos_th: float = 0.001,       # +0.10%
    initial_capital: float = INITIAL_CAPITAL,
) -> np.ndarray:
    """
    Simple long/cash strategy:
      - if pred < neg_th: go to cash (position = 0)
      - if pred > pos_th: go all-in (position = 1)
      - else: keep previous position
    Returns equity curve starting at 'initial_capital'.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    assert y_true.shape == y_pred.shape

    pos = 0.0
    rets: list[float] = []

    for r, p in zip(y_true, y_pred):
        if p < neg_th:
            pos = 0.0
        elif p > pos_th:
            pos = 1.0
        # else: keep previous position

        rets.append(pos * r)

    rets_arr = np.asarray(rets, dtype=float)
    equity = initial_capital * np.cumprod(1.0 + rets_arr)
    return equity


# ---------------------------------------------------------------------
# Main backtest (robust, volatility-scaled)
# ---------------------------------------------------------------------
def run_backtest():
    df, selection_results = load_data_and_results()

    # ------------------------
    # Train / Test split
    # ------------------------
    df_train = df[(df["Date"] >= "2015-01-01") & (df["Date"] <= "2019-12-31")].reset_index(drop=True)
    df_test = df[(df["Date"] >= "2020-01-01") & (df["Date"] <= "2025-12-31")].reset_index(drop=True)

    test_returns = df_test["Return_t"].values

    equities: Dict[str, np.ndarray] = {}
    metrics = []

    for name in TOP_MODELS:
        if name not in selection_results or name not in MODEL_REGISTRY:
            continue

        model_class = MODEL_REGISTRY[name]["class"]
        best_params = selection_results[name]["best_params"]

        model = model_class(**best_params)

        if hasattr(model, "epochs") and model.epochs:
            model.epochs = min(model.epochs, BACKTEST_MAX_EPOCHS)

        X_train, y_train = model.prepare_data(df_train)
        X_test, y_test = model.prepare_data(df_test)

        model.fit(X_train, y_train)

        pred_train = model.predict(X_train).reshape(-1)
        pred_test = model.predict(X_test).reshape(-1)

        # Align predictions
        offset = len(test_returns) - len(pred_test)
        r = test_returns[offset:]
        p = pred_test

        # ------------------------
        # Threshold calibration (TRAIN ONLY)
        # ------------------------
        pos_th = np.percentile(pred_train, 90)
        neg_th = np.percentile(pred_train, 10)

        eq = backtest_strategy(r, p, pos_th, neg_th)
        equities[name] = eq

        metrics.append({
            "strategy": name,
            "CAGR_%": annualized_return(eq) * 100,
            "MaxDD_%": max_drawdown(eq) * 100,
        })

        # ------------------------
        # Random signal sanity check
        # ------------------------
        rand_eq = backtest_strategy(r, np.random.permutation(p), pos_th, neg_th)
        equities[f"{name}_RANDOM"] = rand_eq

    # ------------------------
    # Buy & Hold
    # ------------------------
    bh_eq = np.cumprod(1.0 + test_returns[-len(eq):])
    equities["BUY_AND_HOLD"] = bh_eq

    metrics.append({
        "strategy": "BUY_AND_HOLD",
        "CAGR_%": annualized_return(bh_eq) * 100,
        "MaxDD_%": max_drawdown(bh_eq) * 100,
    })

    metrics_df = pd.DataFrame(metrics).sort_values("CAGR_%", ascending=False)
    print("\nBacktest summary:")
    print(metrics_df.to_string(index=False))

    # ------------------------
    # Plot
    # ------------------------
    dates = df_test["Date"].values[-len(bh_eq):]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare a color palette for model strategies (exclude buy & hold)
    model_names = [n for n in equities.keys() if n != "BUY_AND_HOLD"]
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(model_names), 1)))
    color_map = {name: colors[i] for i, name in enumerate(model_names)}

    for name, eq in equities.items():
        # eq starts at 1.0, so treat as multiple of initial capital
        cum_ret_pct = (eq - 1.0) * 100.0
        if name == "BUY_AND_HOLD":
            label = "STOXX Europe 600 — Buy & Hold"
            ax.plot(
                dates,
                cum_ret_pct,
                label=label,
                color="black",
                linewidth=2.5,
                linestyle="-",
            )
        else:
            label = name
            ax.plot(
                dates,
                cum_ret_pct,
                label=label,
                color=color_map.get(name, None),
                linewidth=1.8,
            )

    ax.set_title(
        "Robust Backtest: Spillover Strategy vs Buy & Hold",
        fontsize=16,
        weight="bold",
        pad=10,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative return (%)", fontsize=12)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)

    ax.legend(ncol=3, fontsize=9)
    plt.tight_layout()

    out = PROJECT_ROOT / "results" / "backtest_robust.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=300)
    plt.close()

    print(f"\nBacktest figure saved to {out}")


# ---------------------------------------------------------------------
# Threshold backtest with 10k EUR initial capital
# ---------------------------------------------------------------------
def run_backtest_top_models():
    df_full, selection_results = load_data_and_results()

    # Backtest starts with 10k EUR
    initial_capital = INITIAL_CAPITAL

    # ------------------------------------------------------------------
    # Train on 2015–2019, test on 2020–2025
    # ------------------------------------------------------------------
    df_full = df_full.sort_values("Date").reset_index(drop=True)

    train_start = pd.Timestamp("2015-01-01")
    train_end = pd.Timestamp("2019-12-31")
    test_start = pd.Timestamp("2020-01-01")
    test_end = pd.Timestamp("2025-12-31")

    df_train = df_full[(df_full["Date"] >= train_start) & (df_full["Date"] <= train_end)].reset_index(drop=True)
    df_test = df_full[(df_full["Date"] >= test_start) & (df_full["Date"] <= test_end)].reset_index(drop=True)

    if df_train.empty or df_test.empty:
        raise ValueError(
            f"[back_test] Empty train/test split with given date ranges: "
            f"train=({train_start}–{train_end}), test=({test_start}–{test_end})"
        )

    test_returns = df_test["Return_t"].values  # underlying asset daily returns for test period

    model_series: Dict[str, Dict[str, np.ndarray]] = {}

    for model_name in TOP_MODELS:
        if model_name not in selection_results or model_name not in MODEL_REGISTRY:
            continue

        info = selection_results[model_name]
        best_params = info["best_params"]
        model_class = MODEL_REGISTRY[model_name]["class"]

        print(f"\n[back_test] Training and predicting with {model_name}...")
        model = model_class(**best_params)

        # Cap epochs for backtest if the model exposes an 'epochs' attribute
        if hasattr(model, "epochs"):
            original_epochs = getattr(model, "epochs")
            if original_epochs is None or original_epochs > BACKTEST_MAX_EPOCHS:
                print(
                    f"[back_test] Capping {model_name} epochs "
                    f"from {original_epochs} to {BACKTEST_MAX_EPOCHS} for backtest."
                )
                model.epochs = BACKTEST_MAX_EPOCHS

        # prepare data using train/test df
        X_train, y_train = model.prepare_data(df_train)
        X_test, y_test = model.prepare_data(df_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Align predictions with test returns
        offset = len(test_returns) - len(y_pred)
        aligned_true = test_returns[offset:]
        aligned_pred = y_pred

        model_series[model_name] = {
            "returns": aligned_true,
            "preds": aligned_pred,
        }

    if not model_series:
        print("[back_test] No models available for backtest.")
        return

    # Determine common horizon across all models
    common_len = min(len(v["returns"]) for v in model_series.values())

    # Use the last common_len days of the test set for buy-and-hold
    test_returns_common = test_returns[-common_len:]
    test_dates_common = df_test["Date"].values[-common_len:]

    # Buy & hold equity (fully invested STOXX Europe 600, starting with 10k EUR)
    bh_equity = initial_capital * np.cumprod(1.0 + test_returns_common)

    equities: Dict[str, np.ndarray] = {"BUY_AND_HOLD": bh_equity}
    for model_name, series in model_series.items():
        r = series["returns"][-common_len:]
        p = series["preds"][-common_len:]
        # use simple threshold strategy here (not volatility-scaled)
        eq = backtest_threshold_strategy(r, p, initial_capital=initial_capital)
        equities[model_name] = eq

    # Compute annualized returns (CAGR) for each curve
    TRADING_DAYS_PER_YEAR = 252.0
    years = common_len / TRADING_DAYS_PER_YEAR if common_len > 0 else 0.0
    annual_returns: Dict[str, float] = {}
    if years > 0:
        for name, eq in equities.items():
            final_equity = float(eq[-1])
            if final_equity <= 0:
                ann = float("nan")
            else:
                # CAGR based on initial_capital -> final_equity
                ann = (final_equity / initial_capital) ** (1.0 / years) - 1.0
            annual_returns[name] = ann * 100.0  # percent

    # Build and save annual-returns table
    ann_table = (
        pd.DataFrame(
            [
                {
                    "Strategy": "STOXX Europe 600 — Buy & Hold" if name == "BUY_AND_HOLD" else name,
                    "CAGR (%)": val,
                }
                for name, val in annual_returns.items()
            ]
        )
        .sort_values("CAGR (%)", ascending=False)
        .reset_index(drop=True)
    )
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    ann_csv = results_dir / "backtest_threshold_strategy_annual_returns.csv"
    ann_table.to_csv(ann_csv, index=False)
    print("\nAnnualized returns (% per year):")
    print(ann_table.to_string(index=False))
    print(f"Annual returns table saved to {ann_csv}")

    # ------------------------------------------------------------------
    # Plot cumulative performance (y-axis in %, x-axis as dates)
    # ------------------------------------------------------------------
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(14, 7))

    # Prepare a color palette for model strategies (exclude buy & hold)
    model_names = [n for n in equities.keys() if n != "BUY_AND_HOLD"]
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(model_names), 1)))
    color_map = {name: colors[i] for i, name in enumerate(model_names)}

    for name, eq in equities.items():
        # convert equity in EUR to cumulative return in percent
        cum_ret_pct = (eq / initial_capital - 1.0) * 100.0
        if name == "BUY_AND_HOLD":
            label = "STOXX Europe 600 — Buy & Hold"
            ax.plot(
                test_dates_common,
                cum_ret_pct,
                label=label,
                color="black",
                linewidth=2.5,
                linestyle="-",
            )
        else:
            label = name
            ax.plot(
                test_dates_common,
                cum_ret_pct,
                label=label,
                color=color_map.get(name, None),
                linewidth=1.8,
            )

    ax.set_title(
        "Backtest: Threshold Strategy vs STOXX Europe 600 Buy & Hold",
        fontsize=16,
        weight="bold",
        pad=10,
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Cumulative return (%)", fontsize=12)

    # nicer date formatting on x-axis
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    # y-axis in percent with 0% reference line
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100.0, decimals=0))
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)

    ax.grid(True, alpha=0.3)

    # Legend at bottom center, outside axes
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=min(len(labels), 3),
        frameon=False,
    )

    # Annualized returns text at bottom-left
    lines = []
    for _, row in ann_table.iterrows():
        val = row["CAGR (%)"]
        if np.isfinite(val):
            s = f"{val: .2f}%"
        else:
            s = "n/a"
        lines.append(f"{row['Strategy']}: {s}")
    text = "\nAnnualized return over test period:\n" + "\n".join(lines)
    fig.text(
        0.01,
        0.11,  # slightly above legend, away from x-axis labels
        fontsize=9,
        va="bottom",
        ha="left",
        s=text,
    )

    # Leave enough bottom margin for legend + text
    plt.tight_layout(rect=(0, 0.20, 1, 1))
    out_path = PROJECT_ROOT / "results" / "backtest_threshold_strategy.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"[back_test] Backtest figure saved to {out_path}")


if __name__ == "__main__":
    run_backtest_top_models()
