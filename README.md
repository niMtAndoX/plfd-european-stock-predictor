# 📈 PLFD European Stock Predictor

## 📝 Description
This project was created for the course **Predictive Learning from Data**  
at **National Taiwan Normal University**, Winter Semester **2025**.

It aims to explore, engineer, and evaluate machine-learning models for financial forecasting.

---

## 🎯 Goal of the Project
The goal is to evaluate multiple prediction models and identify one that can **reliably predict the opening price of the STOXX Europe 600**, using several **Asian stock indices** as input features.

---

## 🚀 How to Use

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/niMtAndoX/plfd-european-stock-predictor.git
cd plfd-european-stock-predictor
```

---

### 2️⃣ Create a Virtual Environment

#### Windows
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

#### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Full Pipeline

The entire workflow is orchestrated by:

```bash
python src/run_full_pipeline.py
```

This script runs, in order:

1. **ETL pipeline**: downloads raw data, builds features and aggregated Asian inputs.
2. **Model selection**: runs cross-validated hyperparameter search for all active models.
3. **Model evaluation**: trains each best model on a training period and evaluates on a held-out test set.
4. **Visualization**: generates paper-ready result plots.

#### 4.1 Controlling which steps run

`run_full_pipeline.py` is controlled via environment variables.  
Each variable is optional; if not set, the default in parentheses is used.

| Env var       | Type   | Default | Meaning                                                                                  |
|---------------|--------|---------|------------------------------------------------------------------------------------------|
| `RUN_ETL`     | bool   | `True`  | If `True`, run the ETL pipeline (`src.data_processing.etl_pipeline`).                   |
| `RUN_SELECTION` | bool | `True`  | If `True`, run model selection / hyperparameter search (`src.model_selection.run_selection`). |
| `RUN_EVAL`    | bool   | `True`  | If `True`, evaluate best models on a test set (`src.evaluation.evaluate_models`).       |
| `RUN_VIZ`     | bool   | `True`  | If `True`, generate result visualizations (`src.evaluation.visualize_results`).         |

Accepted “true” values: `1`, `true`, `yes`, `on` (case-insensitive).  
Accepted “false” values: `0`, `false`, `no`, `off`.

**Examples**

Run the **full** pipeline (default behavior):

```bash
python src/run_full_pipeline.py
```

Run **only** model selection and evaluation (skip ETL and visualization):

```bash
RUN_ETL=0 RUN_VIZ=0 python src/run_full_pipeline.py
```

Run **only** visualization on already existing results:

```bash
RUN_ETL=0 RUN_SELECTION=0 RUN_EVAL=0 RUN_VIZ=1 python src/run_full_pipeline.py
```

On Windows PowerShell, you can set variables like:

```powershell
$env:RUN_ETL = "0"
$env:RUN_VIZ = "0"
python src/run_full_pipeline.py
```

#### 4.2 Active models

Inside `run_full_pipeline.py`, the following models are activated for selection:

```python
active_models = (
    "CNN,SACLSTM,SCINET,RANDOM_FOREST,XGBOOST,"
    "RNN,MTSMFF,DILATED_RNN,"
    "TRANSFORMER,TFT,PYRAFORMER,PREFORMER,AUTOFORMER"
)
```

You can change this list to restrict the search, for example:

```python
active_models = "XGBOOST,RANDOM_FOREST"
```

This string is passed as the `ACTIVE_MODELS` environment variable to the model-selection step.

#### 4.3 Generated outputs

After a successful run you will typically see:

- **ETL outputs** (features, including combined Asian inputs):  
  `src/data_processing/data/clean_features/`
  - `STOXX600_with_asian_features.csv` (STOXX target + aggregated Asian features)
- **Model-selection results**:  
  `results/model_selection_results.json`
- **Evaluation leaderboard**:  
  `results/leaderboard.csv`
- **Plots / figures** (exact paths depend on your visualization module):  
  under `results/` or `src/data_processing/data/img/` depending on configuration.

---

### 5️⃣ Run Backtests

The project provides **two backtest modes** implemented in  
`src/back_test/back_test.py`:

1. **Threshold strategy backtest** (default when running the module)
2. **Robust volatility‑scaled backtest** (optional, called explicitly)

Both backtests assume you have already run:

- the **ETL** step (so `STOXX600_with_asian_features.csv` exists), and  
- the **model selection** step (so `results/model_selection_results.json` exists).

You can get both by running the full pipeline once:

```bash
python src/run_full_pipeline.py
```

or by at least running:

```bash
RUN_ETL=1 RUN_SELECTION=1 RUN_EVAL=0 RUN_VIZ=0 python src/run_full_pipeline.py
```

#### 5.1 Threshold strategy backtest (10k EUR, default)

This backtest:

- trains the **top models** (`XGBOOST`, `RANDOM_FOREST`, `DILATED_RNN`, `SACLSTM`, `MTSMFF`)
  on **2015‑01‑01 → 2019‑12‑31**,
- tests on **2020‑01‑01 → 2025‑12‑31**,
- applies a **simple long/cash threshold rule** per model, starting with **10,000 EUR**, and
- compares each model with a **STOXX Europe 600 buy‑and‑hold** benchmark.

Run it with:

```bash
python -m src.back_test.back_test
```

or from the project root (if `src` is on `PYTHONPATH`):

```bash
python -m src.back_test.back_test
```

Outputs:

- `results/backtest_threshold_strategy_annual_returns.csv`  
  – annualized returns (CAGR %) for each model and for buy‑and‑hold.
- `results/backtest_threshold_strategy.png`  
  – plot of cumulative performance (in %) over the test period.

You can limit training time during backtests via:

```bash
# cap deep-learning models to 10 epochs during backtest
BACKTEST_MAX_EPOCHS=10 python -m src.back_test.back_test
```

If unset, `BACKTEST_MAX_EPOCHS` defaults to **20**.

#### 5.2 Robust volatility‑scaled backtest (optional)

The file also contains a more advanced backtest:

- function: `run_backtest()` in `src/back_test/back_test.py`
- strategy: **volatility‑scaled long/cash** with transaction costs
- returns an equity curve starting at **1.0** for each model and for a buy‑and‑hold baseline.

You can call it without changing the file by using the Python CLI:

```bash
python -c "from src.back_test.back_test import run_backtest; run_backtest()"
```

This produces:

- `results/backtest_robust.png`  
  – comparison of robust strategies vs. STOXX Europe 600 buy‑and‑hold.

It also respects the `BACKTEST_MAX_EPOCHS` environment variable in the same way as the threshold backtest.
