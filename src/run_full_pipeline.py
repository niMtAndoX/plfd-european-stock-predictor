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


def main():
    ROOT = Path(__file__).resolve().parent.parent  # project root (inner folder)

    # Toggle ETL step here (or via env var)
    RUN_ETL = False  # set to True when you want to re-run downloads & feature building

    # 1) ETL PIPELINE
    if RUN_ETL:
        run_module("src.data_processing.etl_pipeline", ROOT)
    else:
        print("⏭ Skipping ETL pipeline (RUN_ETL is False)")

    # 2) MODEL SELECTION (all models except CNN and SACLSTM)
    # Keys must match MODEL_REGISTRY (CNN, SACLSTM, SCINET, RANDOM_FOREST, ...)
    #active_models = (
    #    "SCINET,RANDOM_FOREST,XGBOOST,"
    #   "RNN,MTSMFF,DILATED_RNN,"
    #    "TRANSFORMER,TFT,PYRAFORMER,PREFORMER,AUTOFORMER"
    #)
    active_models ="CNN"
    run_module(
        "src.model_selection.run_selection",
        ROOT,
        extra_env={"ACTIVE_MODELS": active_models},
    )

    # 3) MODEL EVALUATION
    run_module("src.evaluation.evaluate_models", ROOT)

    # 4) VISUALIZATION
    run_module("src.evaluation.visualize_results", ROOT)

    print("\n🎉 FULL PIPELINE FINISHED SUCCESSFULLY!\n")


if __name__ == "__main__":
    main()
