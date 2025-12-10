"""
run_full_pipeline.py

Master script that runs the full workflow:
1. ETL Pipeline
2. Model Selection & Hyperparameter Search
3. Model Evaluation on Test Set
4. Visualization (paper-ready figures)
"""

import subprocess
import sys
from pathlib import Path
import os


def run_module(module_name: str, project_root: Path, extra_env: dict | None = None):
    """Run a module (python -m package.module) from the project root."""
    print(f"\n🔷 Running module: {module_name}\n")
    env = os.environ.copy()
    # Do not override PYTHONPATH; rely on 'src' being a package
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        env=env,
        cwd=project_root,
    )

    if result.returncode != 0:
        print("❌ ERROR in module:", module_name)
        sys.exit(1)

    print(f"✔ Completed: {module_name}\n")


def str_to_bool(val: str | None, default: bool) -> bool:
    """
    Helper to parse boolean env vars.
    Accepts: '1','true','yes','on' (case-insensitive) as True,
             '0','false','no','off' as False,
             None -> default.
    """
    if val is None:
        return default
    v = val.strip().lower()
    if v in {"1", "true", "yes", "on"}:
        return True
    if v in {"0", "false", "no", "off"}:
        return False
    return default


def main():
    ROOT = Path(__file__).resolve().parent.parent  # project root (inner folder)

    # ------------------------------------------------------------------
    # Toggle steps here or via env vars:
    #   RUN_ETL, RUN_SELECTION, RUN_EVAL, RUN_VIZ
    # ------------------------------------------------------------------
    RUN_ETL = str_to_bool(os.getenv("RUN_ETL"), False)
    RUN_SELECTION = str_to_bool(os.getenv("RUN_SELECTION"), False)
    RUN_EVAL = str_to_bool(os.getenv("RUN_EVAL"), False)
    RUN_VIZ = str_to_bool(os.getenv("RUN_VIZ"), True)

    # 1) ETL PIPELINE
    if RUN_ETL:
        run_module("src.data_processing.etl_pipeline", ROOT)
    else:
        print("⏭ Skipping ETL pipeline (RUN_ETL is False)")

    # 2) MODEL SELECTION
    # Keys must match MODEL_REGISTRY (CNN, SACLSTM, SCINET, RANDOM_FOREST, ...)
    active_models = (
        "CNN,SACLSTM,SCINET,RANDOM_FOREST,XGBOOST,"
        "RNN,MTSMFF,DILATED_RNN,"
        "TRANSFORMER,TFT,PYRAFORMER,PREFORMER,AUTOFORMER"
    )
    if RUN_SELECTION:
        run_module(
            "src.model_selection.run_selection",
            ROOT,
            extra_env={"ACTIVE_MODELS": active_models},
        )
    else:
        print("⏭ Skipping model selection (RUN_SELECTION is False)")

    # 3) MODEL EVALUATION
    if RUN_EVAL:
        run_module("src.evaluation.evaluate_models", ROOT)
    else:
        print("⏭ Skipping model evaluation (RUN_EVAL is False)")

    # 4) VISUALIZATION
    if RUN_VIZ:
        run_module("src.evaluation.visualize_results", ROOT)
    else:
        print("⏭ Skipping visualization (RUN_VIZ is False)")

    print("\n🎉 FULL PIPELINE FINISHED SUCCESSFULLY!\n")


if __name__ == "__main__":
    main()
