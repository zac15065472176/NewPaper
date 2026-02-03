"""
Base experiment class for AdaPrune experiments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import time

from adaprune.utils.data_loader import DataLoader


@dataclass
class BaseExperiment:
    experiment_name: str
    datasets: list
    n_folds: int = 5
    random_state: int = 42
    results_dir: str = "results"
    verbose: int = 1

    def setup(self):
        os.makedirs(self.results_dir, exist_ok=True)

    def load_datasets(self) -> Dict[str, Any]:
        loader = DataLoader()
        return {name: loader.load_dataset(name) for name in self.datasets}

    def get_models(self) -> Dict[str, Any]:
        raise NotImplementedError

    def run_single_dataset(self, name, X, y) -> pd.DataFrame:
        results = []
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        models = self.get_models()

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for mname, model in models.items():
                start = time.time()
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                train_time = time.time() - start

                auc = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)
                    try:
                        auc = roc_auc_score(y_test, proba, multi_class="ovr")
                    except Exception:
                        auc = None

                results.append(
                    {
                        "dataset": name,
                        "fold": fold,
                        "method": mname,
                        "accuracy": accuracy_score(y_test, preds),
                        "f1": f1_score(y_test, preds, average="weighted"),
                        "auc": auc,
                        "train_time": train_time,
                    }
                )
        return pd.DataFrame(results)

    def run(self) -> pd.DataFrame:
        self.setup()
        data = self.load_datasets()
        all_results = []
        for name, (X, y, _) in data.items():
            if self.verbose:
                print(f"Running dataset: {name}")
            all_results.append(self.run_single_dataset(name, X, y))
        results = pd.concat(all_results, ignore_index=True)
        self.save_results(results)
        summary = self.generate_summary(results)
        summary.to_csv(os.path.join(self.results_dir, f"{self.experiment_name}_summary.csv"), index=False)
        return results

    def save_results(self, results: pd.DataFrame):
        path = os.path.join(self.results_dir, f"{self.experiment_name}_results.csv")
        results.to_csv(path, index=False)

    def generate_summary(self, results: pd.DataFrame) -> pd.DataFrame:
        return results.groupby(["method"])[["accuracy", "f1", "auc", "train_time"]].mean().reset_index()