"""
Efficiency analysis experiment (Optimized Callback Version).
"""

from __future__ import annotations

import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

# Replace pure Python GBDT with the optimized Native Callback XGB implementation
from adaprune.core.adaprune_xgb import AdaPruneXGB


def main():
    sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    results = []

    for n in sizes:
        # Generate synthetic binary classification dataset
        X, y = make_classification(n_samples=n, n_features=30, random_state=42)

        # 1. Test AdaPrune (Optimized C++ Callback)
        model = AdaPruneXGB(n_estimators=50, verbose=0)
        start = time.time()
        model.fit(X, y)
        t_adaprune = time.time() - start

        # 2. Test standard XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=50, eval_metric="logloss", use_label_encoder=False, verbosity=0)
        start = time.time()
        xgb_model.fit(X, y)
        t_xgb = time.time() - start

        # 3. Test standard Grid Search (Traditional HPO)
        grid = GridSearchCV(
            xgb.XGBClassifier(n_estimators=50, eval_metric="logloss", use_label_encoder=False, verbosity=0),
            param_grid={"max_depth": [3, 6], "min_child_weight": [1, 3]},
            cv=3,
        )
        start = time.time()
        grid.fit(X, y)
        t_grid = time.time() - start

        overhead = (t_adaprune - t_xgb) / max(t_xgb, 1e-6) * 100

        results.append(
            {
                "n_samples": n,
                "time_adaprune": round(t_adaprune, 4),
                "time_xgboost": round(t_xgb, 4),
                "time_gridsearch": round(t_grid, 4),
                "overhead_percent": round(overhead, 2),
            }
        )

    # Output formatted results
    print(f"{'Samples':<10} | {'AdaPrune(s)':<12} | {'XGBoost(s)':<12} | {'GridSearch(s)':<14} | {'Overhead(%)':<12}")
    print("-" * 70)
    for r in results:
        print(f"{r['n_samples']:<10} | {r['time_adaprune']:<12.4f} | {r['time_xgboost']:<12.4f} | {r['time_gridsearch']:<14.4f} | {r['overhead_percent']:<12.2f}%")


if __name__ == "__main__":
    main()