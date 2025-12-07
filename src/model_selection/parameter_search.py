# src/model_selection/parameter_search.py

"""
Grid search over each model's hyperparameter space using K-fold CV.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple

import itertools
import json
import os
from pathlib import Path

from .model_registry import MODEL_REGISTRY
from .cross_validation import cross_validate_on_dataframe


def param_grid(search_space: Dict[str, list]):
    """Yield dicts for each combination in the search space."""
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def search_best_params_for_model(
    model_name: str,
    model_info: Dict[str, Any],
    df,
    k_folds: int = 5,
) -> Tuple[Dict[str, Any], float]:
    model_class = model_info["class"]
    search_space = model_info["search"]

    best_params = None
    best_score = float("inf")

    for params in param_grid(search_space):
        print(f"\n[{model_name}] Testing params: {params}")
        mean_rmse = cross_validate_on_dataframe(model_class, params, df, k=k_folds)

        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params
            print(f"  -> New best for {model_name}: RMSE={best_score:.6f}")

    if best_params is None:
        raise RuntimeError(f"No params evaluated for model {model_name}")

    return best_params, best_score


def run_full_model_selection(
    df,
    k_folds: int = 5,
    output_path: str | Path | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Run hyperparameter search for ALL (or a subset of) models in the registry.

    Returns:
        results dict: {model_name: {"best_params":{...}, "cv_rmse": float}}
    """
    # Optional filter via env var, e.g. ACTIVE_MODELS="CNN,RNN"
    active = os.environ.get("ACTIVE_MODELS")
    if active:
        active_set = {name.strip().upper() for name in active.split(",")}
        model_items = {
            name: info
            for name, info in MODEL_REGISTRY.items()
            if name.upper() in active_set
        }
    else:
        model_items = MODEL_REGISTRY

    print("[parameter_search] Starting full model selection")
    print(f"[parameter_search] Models to run: {list(model_items.keys())}")

    results: Dict[str, Dict[str, Any]] = {}

    for model_name, info in model_items.items():
        print(f"\n=== Searching best parameters for {model_name} ===")
        best_params, best_rmse = search_best_params_for_model(
            model_name, info, df, k_folds=k_folds
        )
        # Show which model+params were finally trained/evaluated as best
        print(f"[{model_name}] Finished search. Best params: {best_params}")
        results[model_name] = {
            "best_params": best_params,
            "cv_rmse": best_rmse,
        }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved model selection results to {output_path}")

    return results
