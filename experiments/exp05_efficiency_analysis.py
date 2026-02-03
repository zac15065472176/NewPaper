"""
Efficiency analysis experiment.
"""

from __future__ import annotations

import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification

from adaprune.core.adaprune_gbdt import AdaPruneGBDT


def main():
    sizes = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    results = []

    for n in sizes:
        X, y = make_classification(n_samples=n, n_features=30, random_state=42)

        model = AdaPruneGBDT(n_estimators=50)
        start = time.time()
        model.fit(X, y)
        t_adaprune = time.time() - start

        xgb_model = xgb.XGBClassifier(eval_metric="logloss")
        start = time.time()
        xgb_model.fit(X, y)
        t_xgb = time.time() - start

        grid = GridSearchCV(
            xgb.XGBClassifier(eval_metric="logloss"),
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
                "time_adaprune": t_adaprune,
                "time_xgboost": t_xgb,
                "time_gridsearch": t_grid,
                "overhead_percent": overhead,
            }
        )

    for r in results:
        print(r)


if __name__ == "__main__":
    main()