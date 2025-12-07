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

    # Let child write directly to our stdout/stderr (no buffering in Python)
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

    # 1) ETL PIPELINE
    run_module("src.data_processing.etl_pipeline", ROOT)

    # 2) MODEL SELECTION (all models)
    run_module("src.model_selection.run_selection", ROOT)

    # 3) MODEL EVALUATION
    run_module("src.evaluation.evaluate_models", ROOT)

    # 4) VISUALIZATION
    run_module("src.evaluation.visualize_results", ROOT)

    print("\n🎉 FULL PIPELINE FINISHED SUCCESSFULLY!\n")


if __name__ == "__main__":
    main()
