"""
Sensitivity analysis experiment.
"""

from __future__ import annotations

from itertools import product
import pandas as pd
from adaprune.core.adaprune_gbdt import AdaPruneGBDT
from adaprune.utils.visualization import plot_learning_curves
from adaprune.utils.data_loader import DataLoader


def main():
    loader = DataLoader()
    datasets = ["adult", "diabetes", "synthetic_noisy_small"]

    adaptation_frequency = [1, 3, 5, 10, 20, 50]
    gap_threshold = [0.01, 0.03, 0.05, 0.1, 0.2]
    depth_step = [1, 2, 3]

    results = []

    for name in datasets:
        X, y, _ = loader.load_dataset(name)
        for af, gt, ds in product(adaptation_frequency, gap_threshold, depth_step):
            model = AdaPruneGBDT(
                strategy="gap_based",
                adaptation_frequency=af,
            )
            model.fit(X, y)
            history = model.get_adaptation_history()
            results.append(
                {
                    "dataset": name,
                    "adaptation_frequency": af,
                    "gap_threshold": gt,
                    "depth_step": ds,
                    "final_train_loss": history["state_history"][-1]["train_loss"],
                    "final_val_loss": history["state_history"][-1]["val_loss"],
                }
            )
    df = pd.DataFrame(results)
    df.to_csv("results/exp04_sensitivity_results.csv", index=False)
    print("Saved exp04_sensitivity_results.csv")


if __name__ == "__main__":
    main()