"""
严谨的参数灵敏性分析脚本 (基于 Gap-Based 策略)
运行方式: 在项目根目录下执行 python run_sensitivity.py
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# 确保能引入 adaprune
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from adaprune.core.adaprune_gbdt import AdaPruneGBDT
import adaprune.core.pruning_controller as pc

DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'

def load_dataset(name):
    file_path = DATA_DIR / f"{name}.pkl"
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y']

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 【改动点】换成 synthetic_noisy，并增加阈值点
    datasets = ['ionosphere', 'synthetic_noisy_small']
    thresholds = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]

    all_results = {}

    # 备份原始的 init
    orig_init = pc.GapBasedStrategy.__init__

    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Running sensitivity analysis on: {dataset_name}")
        print(f"{'='*50}")

        try:
            X, y = load_dataset(dataset_name)
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            continue

        dataset_accs = []
        dataset_gaps = []

        for th in thresholds:
            print(f"  Testing gap_threshold = {th} ...")

            # 使用猴子补丁强行修改 GapBasedStrategy 的默认 gap_threshold
            def patched_init(self, gap_threshold=th, trend_threshold=0.001, depth_step=1, samples_factor=1.5):
                orig_init(self, gap_threshold=gap_threshold, trend_threshold=trend_threshold, depth_step=depth_step, samples_factor=samples_factor)

            pc.GapBasedStrategy.__init__ = patched_init

            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_accs = []
            fold_gaps = []

            for train_idx, test_idx in kfold.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                model = AdaPruneGBDT(
                    n_estimators=100,
                    strategy='gap_based',
                    adaptation_frequency=5,
                    random_state=42,
                    verbose=0
                )

                model.fit(X_train, y_train)

                y_pred_test = model.predict(X_test)
                y_pred_train = model.predict(X_train)

                test_acc = accuracy_score(y_test, y_pred_test)
                train_acc = accuracy_score(y_train, y_pred_train)
                gap = train_acc - test_acc

                fold_accs.append(test_acc)
                fold_gaps.append(gap)

            mean_acc = np.mean(fold_accs)
            mean_gap = np.mean(fold_gaps)

            print(f"    -> Acc: {mean_acc:.4f} | Gap: {mean_gap:.4f}")
            dataset_accs.append(mean_acc)
            dataset_gaps.append(mean_gap)

        all_results[dataset_name] = {
            'acc': dataset_accs,
            'gap': dataset_gaps
        }

    # 恢复原始 init
    pc.GapBasedStrategy.__init__ = orig_init

    # ==========================
    # 开始画图
    # ==========================
    print("\nGenerating Figure...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, dataset in enumerate(datasets):
        if dataset not in all_results:
            continue

        ax = axes[i]
        acc = all_results[dataset]['acc']
        gap = all_results[dataset]['gap']

        color1 = '#3274A1'
        ax.plot(thresholds, acc, marker='o', linewidth=2, color=color1, label='Accuracy')
        ax.set_xlabel('Gap Threshold ($\\theta_g$)', fontsize=11)
        ax.set_ylabel('Test Accuracy', color=color1, fontsize=11)
        ax.tick_params(axis='y', labelcolor=color1)
        ax.set_title(f"Sensitivity on {dataset}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        # 阴影区标示稳定发挥的区间（通用化）
        ax.axvspan(0.03, 0.09, color='gray', alpha=0.15, label='Stable Region')

        ax_twin = ax.twinx()
        color2 = '#E1812C'
        ax_twin.plot(thresholds, gap, marker='s', linestyle='--', linewidth=2, color=color2, label='Generalization Gap')
        ax_twin.set_ylabel('Generalization Gap', color=color2, fontsize=11)
        ax_twin.tick_params(axis='y', labelcolor=color2)

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left')

    plt.tight_layout()
    save_path = RESULTS_DIR / 'Figure6_Sensitivity.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nDone! Figure saved to {save_path}")

if __name__ == "__main__":
    main()