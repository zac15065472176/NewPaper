"""
实验2: 过拟合场景分析

分析AdaPrune在不同过拟合风险场景下的表现
"""

import os
import sys
import time
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from tqdm import tqdm

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adaprune import AdaPruneGBDT
from adaprune.utils import load_dataset

RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
FIGURES_DIR = PROJECT_ROOT / 'results' / 'figures'
RESULTS_DIR. mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_scenario_experiment(n_runs: int = 5, random_state: int = 42):
    """运行场景实验"""
    print("=" * 70)
    print("Experiment 2: Overfitting Scenarios Analysis")
    print("=" * 70)
    
    scenarios = [
        'synthetic_clean_large',
        'synthetic_clean_small',
        'synthetic_noisy_large',
        'synthetic_noisy_small',
        'synthetic_high_dim',
        'synthetic_imbalanced',
    ]
    
    models_config = {
        'XGB_default': lambda: XGBClassifier(n_estimators=100, verbosity=0),
        'XGB_strong_prune': lambda: XGBClassifier(
            n_estimators=100, max_depth=3, min_child_weight=10, verbosity=0
        ),
        'XGB_weak_prune': lambda: XGBClassifier(
            n_estimators=100, max_depth=15, min_child_weight=1, verbosity=0
        ),
        'AdaPrune_hybrid': lambda: AdaPruneGBDT(
            n_estimators=100, strategy='hybrid', verbose=0
        ),
        'AdaPrune_data': lambda: AdaPruneGBDT(
            n_estimators=100, strategy='data_aware', verbose=0
        ),
    }
    
    results = []
    adaptation_histories = {}
    
    for scenario in tqdm(scenarios, desc="Scenarios"):
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario}")
        print(f"{'='*50}")
        
        try:
            X, y, metadata = load_dataset(scenario)
            print(f"  Shape:  {X.shape}, Classes: {len(np.unique(y))}")
        except Exception as e: 
            print(f"  Error:  {e}")
            continue
        
        for run in range(n_runs):
            seed = random_state + run
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )
            
            for model_name, model_fn in models_config. items():
                try:
                    model = model_fn()
                    if hasattr(model, 'random_state'):
                        model.random_state = seed
                    
                    start_time = time. time()
                    model.fit(X_train, y_train)
                    train_time = time. time() - start_time
                    
                    # 训练集和测试集准确率
                    train_acc = accuracy_score(y_train, model.predict(X_train))
                    test_acc = accuracy_score(y_test, model.predict(X_test))
                    gap = train_acc - test_acc
                    
                    results.append({
                        'scenario': scenario,
                        'run': run,
                        'method':  model_name,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'gap': gap,
                        'train_time': train_time,
                    })
                    
                    # 保存AdaPrune的自适应历史
                    if hasattr(model, 'get_adaptation_history') and run == 0:
                        key = f"{scenario}_{model_name}"
                        adaptation_histories[key] = model. get_adaptation_history()
                    
                except Exception as e:
                    print(f"  Error {model_name}:  {e}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(RESULTS_DIR / f'exp02_scenario_results_{timestamp}. csv', index=False)
    
    # 生成汇总
    summary = results_df.groupby(['scenario', 'method']).agg({
        'test_accuracy': ['mean', 'std'],
        'gap':  ['mean', 'std'],
    }).round(4)
    
    print("\n" + "=" * 70)
    print("Scenario Results Summary")
    print("=" * 70)
    
    # 每个场景的最佳方法
    for scenario in scenarios:
        scenario_data = results_df[results_df['scenario'] == scenario]
        best = scenario_data.groupby('method')['test_accuracy']. mean().idxmax()
        best_acc = scenario_data.groupby('method')['test_accuracy'].mean().max()
        print(f"  {scenario}: Best={best} ({best_acc:.4f})")
    
    # 可视化
    try:
        plot_scenario_results(results_df, adaptation_histories)
    except Exception as e: 
        print(f"Visualization error: {e}")
    
    return results_df, summary


def plot_scenario_results(results_df, adaptation_histories):
    """可视化场景结果"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. 热力图
    pivot = results_df.pivot_table(
        values='test_accuracy',
        index='scenario',
        columns='method',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
    ax.set_title('Test Accuracy across Scenarios')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp02_scenario_heatmap.png', dpi=150)
    plt.close()
    
    # 2. Gap对比
    fig, ax = plt.subplots(figsize=(12, 6))
    gap_pivot = results_df. pivot_table(
        values='gap',
        index='scenario',
        columns='method',
        aggfunc='mean'
    )
    gap_pivot.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Generalization Gap')
    ax.set_title('Generalization Gap across Scenarios')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt. xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exp02_gap_comparison.png', dpi=150)
    plt.close()
    
    print(f"Figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_scenario_experiment()