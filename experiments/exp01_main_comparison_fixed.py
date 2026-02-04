"""
实验1: 主实验 - 方法对比（修正版，包含Gap计算）
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 尝试导入AdaPrune，如果失败则跳过
try:
    from adaprune.core.adaprune_gbdt import AdaPruneGBDT
    HAS_ADAPRUNE = True
except ImportError:
    print("Warning: AdaPrune not available, using only baselines")
    HAS_ADAPRUNE = False


def generate_datasets():
    """生成/加载数据集"""
    datasets = {}

    # 使用sklearn自带数据集和合成数据
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer

    # 真实数据集
    print("Loading datasets...")

    # Breast Cancer
    data = load_breast_cancer()
    datasets['breast_cancer'] = (data.data.astype(np.float32), data.target.astype(np.int32))

    # Wine
    data = load_wine()
    datasets['wine'] = (data.data.astype(np.float32), data.target.astype(np.int32))

    # Iris
    data = load_iris()
    datasets['iris'] = (data.data.astype(np.float32), data.target.astype(np.int32))

    # 合成数据集 - 简单
    X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                               n_redundant=3, random_state=42)
    datasets['synthetic_easy'] = (X.astype(np.float32), y.astype(np.int32))

    # 合成数据集 - 噪声
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=5, flip_y=0.15, random_state=42)
    X = X + np.random.randn(*X.shape) * 0.3
    datasets['synthetic_noisy'] = (X.astype(np.float32), y.astype(np.int32))

    # 合成数据集 - 小样本
    X, y = make_classification(n_samples=300, n_features=20, n_informative=15,
                               n_redundant=3, random_state=42)
    datasets['synthetic_small'] = (X.astype(np.float32), y.astype(np.int32))

    # 合成数据集 - 高维
    X, y = make_classification(n_samples=500, n_features=100, n_informative=20,
                               n_redundant=20, random_state=42)
    datasets['synthetic_highdim'] = (X.astype(np.float32), y.astype(np.int32))

    for name, (X, y) in datasets.items():
        print(f"  {name}: {X.shape}, classes={len(np.unique(y))}")

    return datasets


def get_models():
    """获取所有模型"""
    models = {
        'XGB_default': lambda: XGBClassifier(n_estimators=100, random_state=42,
                                              eval_metric='logloss', verbosity=0),
        'XGB_tuned': lambda: XGBClassifier(n_estimators=100, max_depth=6, min_child_weight=3,
                                            gamma=0.1, subsample=0.8, random_state=42,
                                            eval_metric='logloss', verbosity=0),
        'LGBM_default': lambda: LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'RF_default': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    }

    if HAS_ADAPRUNE:
        models.update({
            'AdaPrune_gap': lambda: AdaPruneGBDT(n_estimators=100, strategy='gap_based',
                                                  adaptation_frequency=5, random_state=42, verbose=0),
            'AdaPrune_trend': lambda: AdaPruneGBDT(n_estimators=100, strategy='trend_based',
                                                    adaptation_frequency=5, random_state=42, verbose=0),
            'AdaPrune_data': lambda: AdaPruneGBDT(n_estimators=100, strategy='data_aware',
                                                   adaptation_frequency=5, random_state=42, verbose=0),
            'AdaPrune_hybrid': lambda: AdaPruneGBDT(n_estimators=100, strategy='hybrid',
                                                     adaptation_frequency=5, random_state=42, verbose=0),
        })

    return models


def run_experiment(n_folds=5, random_state=42):
    """运行实验"""
    print("=" * 70)
    print("Experiment 1: Main Comparison (with Gap)")
    print("=" * 70)

    datasets = generate_datasets()
    models = get_models()

    print(f"\nModels: {list(models.keys())}")
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Folds: {n_folds}")

    results = []

    for dataset_name, (X, y) in tqdm(datasets.items(), desc="Datasets"):
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name} (n={len(y)}, d={X.shape[1]})")

        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for model_name, model_fn in models.items():
                try:
                    model = model_fn()

                    start = time.time()
                    model.fit(X_train, y_train)
                    train_time = time.time() - start

                    # 计算训练集和测试集准确率
                    train_acc = accuracy_score(y_train, model.predict(X_train))
                    test_acc = accuracy_score(y_test, model.predict(X_test))
                    gap = train_acc - test_acc  # Gap = 训练准确率 - 测试准确率

                    f1 = f1_score(y_test, model.predict(X_test), average='weighted')

                    results.append({
                        'dataset': dataset_name,
                        'fold': fold,
                        'method': model_name,
                        'train_accuracy': round(train_acc, 4),
                        'test_accuracy': round(test_acc, 4),
                        'gap': round(gap, 4),
                        'f1': round(f1, 4),
                        'train_time': round(train_time, 4),
                    })

                except Exception as e:
                    print(f"  Error {model_name}: {e}")

        # 打印该数据集的结果
        dataset_results = [r for r in results if r['dataset'] == dataset_name and r['fold'] == 0]
        print(f"  Fold 0 results:")
        for r in sorted(dataset_results, key=lambda x: -x['test_accuracy']):
            print(f"    {r['method']:18s}: Acc={r['test_accuracy']:.4f}, Gap={r['gap']:.4f}")

    # 保存结果
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(RESULTS_DIR / f'exp01_full_results_{timestamp}.csv', index=False)

    # 生成汇总
    summary = results_df.groupby('method').agg({
        'test_accuracy': ['mean', 'std'],
        'gap': ['mean', 'std'],
        'f1': 'mean',
        'train_time': 'mean'
    }).round(4)

    summary.columns = ['acc_mean', 'acc_std', 'gap_mean', 'gap_std', 'f1_mean', 'time_mean']
    summary = summary.sort_values('acc_mean', ascending=False)
    summary.to_csv(RESULTS_DIR / f'exp01_summary_{timestamp}.csv')

    # 按数据集汇总
    by_dataset = results_df.groupby(['dataset', 'method']).agg({
        'test_accuracy': 'mean',
        'gap': 'mean',
    }).round(4).unstack()
    by_dataset.to_csv(RESULTS_DIR / f'exp01_by_dataset_{timestamp}.csv')

    print("\n" + "=" * 70)
    print("Results Summary (by Method)")
    print("=" * 70)
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("Gap Comparison")
    print("=" * 70)
    gap_summary = results_df.groupby('method')['gap'].mean().sort_values()
    print(gap_summary.to_string())

    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"  - exp01_full_results_{timestamp}.csv")
    print(f"  - exp01_summary_{timestamp}.csv")
    print(f"  - exp01_by_dataset_{timestamp}.csv")

    return results_df, summary


if __name__ == "__main__":
    run_experiment()