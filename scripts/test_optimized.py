"""
测试优化版 AdaPruneXGB V2
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import time
import pickle

PROJECT_ROOT = Path(__file__).parent. parent
sys.path.insert(0, str(PROJECT_ROOT))

from adaprune import AdaPruneXGB, AdaPruneGBDT

DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_dataset(name):
    with open(DATA_DIR / f"{name}.pkl", 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y']


def test_synthetic():
    """测试合成数据"""
    print("=" * 70)
    print("Test 1: Synthetic Data Comparison")
    print("=" * 70)
    
    scenarios = {
        'Clean (easy)': {'n_samples': 2000, 'noise': 0.0, 'flip_y': 0.0},
        'Noisy (hard)': {'n_samples': 2000, 'noise': 0.3, 'flip_y': 0.1},
        'Small sample': {'n_samples': 500, 'noise': 0.1, 'flip_y': 0.05},
        'High dim': {'n_samples': 1000, 'n_features': 100, 'noise': 0.1, 'flip_y': 0.05},
    }
    
    results = []
    
    for scenario_name, config in scenarios.items():
        print(f"\n--- {scenario_name} ---")
        
        n_features = config. get('n_features', 20)
        X, y = make_classification(
            n_samples=config['n_samples'],
            n_features=n_features,
            n_informative=min(15, n_features - 2),
            n_redundant=3,
            flip_y=config['flip_y'],
            random_state=42
        )
        
        if config['noise'] > 0:
            np.random.seed(42)
            X = X + np.random.randn(*X.shape) * config['noise']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        models = {
            'XGB_default': XGBClassifier(
                n_estimators=200, verbosity=0, random_state=42, use_label_encoder=False
            ),
            'XGB_tuned': XGBClassifier(
                n_estimators=200, max_depth=5, min_child_weight=3,
                verbosity=0, random_state=42, use_label_encoder=False
            ),
            'XGB_early_stop': XGBClassifier(
                n_estimators=300, early_stopping_rounds=30,
                verbosity=0, random_state=42, use_label_encoder=False
            ),
            'AdaPruneXGB': AdaPruneXGB(
                n_estimators=300, 
                initial_max_depth=6,
                gap_threshold=0.03,
                warmup_rounds=30,
                random_state=42, 
                verbose=0
            ),
        }
        
        for model_name, model in models.items():
            start = time.time()
            
            if 'early_stop' in model_name: 
                model.fit(X_train, y_train, 
                         eval_set=[(X_test, y_test)], verbose=False)
            else:
                model.fit(X_train, y_train)
            
            train_time = time.time() - start
            
            train_acc = accuracy_score(y_train, model.predict(X_train))
            test_acc = accuracy_score(y_test, model.predict(X_test))
            gap = train_acc - test_acc
            
            results.append({
                'scenario': scenario_name,
                'model': model_name,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'gap': gap,
                'time': train_time
            })
            
            print(f"  {model_name:15s}: Test={test_acc:.4f}, Gap={gap:.4f}, Time={train_time:.2f}s")
    
    return pd.DataFrame(results)


def test_real_datasets():
    """测试真实数据集"""
    print("\n" + "=" * 70)
    print("Test 2: Real Datasets Comparison")
    print("=" * 70)
    
    datasets = ['diabetes', 'ionosphere', 'credit-g', 'spambase', 'vehicle', 
                'sonar', 'segment', 'waveform']
    
    # 检查哪些数据集可用
    available = []
    for name in datasets:
        if (DATA_DIR / f"{name}. pkl").exists():
            available. append(name)
    
    if not available:
        print("No datasets available.  Run download_datasets.py first.")
        return None
    
    print(f"Testing on {len(available)} datasets: {available}")
    
    results = []
    
    for dataset_name in available:
        print(f"\n--- {dataset_name} ---")
        
        X, y = load_dataset(dataset_name)
        
        # 3折交叉验证
        kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        models_config = {
            'XGB_default': lambda: XGBClassifier(
                n_estimators=200, verbosity=0, random_state=42, use_label_encoder=False
            ),
            'XGB_tuned': lambda: XGBClassifier(
                n_estimators=200, max_depth=5, min_child_weight=3,
                verbosity=0, random_state=42, use_label_encoder=False
            ),
            'AdaPruneXGB': lambda: AdaPruneXGB(
                n_estimators=300,
                initial_max_depth=6,
                gap_threshold=0.03,
                warmup_rounds=30,
                random_state=42,
                verbose=0
            ),
        }
        
        for model_name, model_fn in models_config.items():
            fold_results = []
            
            for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model = model_fn()
                
                start = time.time()
                model.fit(X_train, y_train)
                train_time = time.time() - start
                
                train_acc = accuracy_score(y_train, model.predict(X_train))
                test_acc = accuracy_score(y_test, model.predict(X_test))
                
                fold_results.append({
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'gap': train_acc - test_acc,
                    'time': train_time
                })
            
            avg_test = np.mean([r['test_acc'] for r in fold_results])
            avg_gap = np.mean([r['gap'] for r in fold_results])
            avg_time = np.mean([r['time'] for r in fold_results])
            
            results. append({
                'dataset': dataset_name,
                'model': model_name,
                'test_acc': avg_test,
                'gap': avg_gap,
                'time': avg_time
            })
            
            print(f"  {model_name: 15s}: Test={avg_test:.4f}, Gap={avg_gap:.4f}, Time={avg_time:.2f}s")
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("AdaPrune XGB V2 - Optimized Version Test")
    print("=" * 70)
    
    # 测试合成数据
    synthetic_results = test_synthetic()
    
    # 测试真实数据
    real_results = test_real_datasets()
    
    # 汇总
    print("\n" + "=" * 70)
    print("Summary: Synthetic Data")
    print("=" * 70)
    if synthetic_results is not None:
        summary = synthetic_results.groupby('model').agg({
            'test_acc': 'mean',
            'gap': 'mean',
            'time': 'mean'
        }).round(4)
        summary = summary.sort_values('test_acc', ascending=False)
        print(summary. to_string())
        
        # 找出各场景最佳
        print("\nBest per scenario:")
        for scenario in synthetic_results['scenario'].unique():
            scenario_data = synthetic_results[synthetic_results['scenario'] == scenario]
            best = scenario_data.loc[scenario_data['test_acc'].idxmax()]
            print(f"  {scenario}: {best['model']} ({best['test_acc']:.4f})")
    
    print("\n" + "=" * 70)
    print("Summary: Real Datasets")
    print("=" * 70)
    if real_results is not None and len(real_results) > 0:
        summary = real_results.groupby('model').agg({
            'test_acc': 'mean',
            'gap': 'mean',
            'time': 'mean'
        }).round(4)
        summary = summary.sort_values('test_acc', ascending=False)
        print(summary. to_string())
        
        # 统计胜出次数
        print("\nWins per model:")
        for model in real_results['model'].unique():
            wins = 0
            for dataset in real_results['dataset'].unique():
                dataset_data = real_results[real_results['dataset'] == dataset]
                best_model = dataset_data.loc[dataset_data['test_acc'].idxmax(), 'model']
                if best_model == model:
                    wins += 1
            print(f"  {model}: {wins}/{len(real_results['dataset']. unique())} datasets")
    
    print("\n" + "=" * 70)
    print("✓ Test completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()