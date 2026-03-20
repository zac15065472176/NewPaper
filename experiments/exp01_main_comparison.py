"""
实验1:  主实验 - 方法对比

对比AdaPrune与基线方法在多个数据集上的表现
"""

import os
import sys
import time
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn. ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adaprune import AdaPruneGBDT

# 数据和结果目录
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'tables'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset(name):
    """加载数据集"""
    file_path = DATA_DIR / f"{name}.pkl"
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data. get('metadata', {})


def list_datasets():
    """列出所有数据集"""
    return [f. stem for f in DATA_DIR.glob("*.pkl")]


def get_baselines():
    """获取基线模型"""
    return {
        'RF_default': lambda:  RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1
        ),
        'XGB_default': lambda: XGBClassifier(
            n_estimators=100, random_state=42, verbosity=0, use_label_encoder=False
        ),
        'XGB_tuned': lambda: XGBClassifier(
            n_estimators=100, max_depth=6, min_child_weight=3,
            learning_rate=0.1, random_state=42, verbosity=0, use_label_encoder=False
        ),
        'LGBM_default': lambda: LGBMClassifier(
            n_estimators=100, random_state=42, verbosity=-1
        ),
    }


def get_adaprune_models():
    """获取AdaPrune模型变体"""
    return {
        'AdaPrune_gap':  lambda: AdaPruneGBDT(
            n_estimators=100, strategy='gap_based',
            adaptation_frequency=5, random_state=42, verbose=0
        ),
        'AdaPrune_trend':  lambda: AdaPruneGBDT(
            n_estimators=100, strategy='trend_based',
            adaptation_frequency=5, random_state=42, verbose=0
        ),
        'AdaPrune_data': lambda: AdaPruneGBDT(
            n_estimators=100, strategy='data_aware',
            adaptation_frequency=5, random_state=42, verbose=0
        ),
        'AdaPrune_hybrid': lambda: AdaPruneGBDT(
            n_estimators=100, strategy='hybrid',
            adaptation_frequency=5, random_state=42, verbose=0
        ),
    }


# def evaluate_model(model, X_train, y_train, X_test, y_test):
#     """评估单个模型"""
#     start_time = time. time()
#     model.fit(X_train, y_train)
#     train_time = time. time() - start_time
#
#     y_pred = model.predict(X_test)
#
#     if hasattr(model, 'predict_proba'):
#         try:
#             y_proba = model.predict_proba(X_test)
#         except:
#             y_proba = None
#     else:
#         y_proba = None
#
#     # 计算指标
#     accuracy = accuracy_score(y_test, y_pred)
#
#     n_classes = len(np.unique(y_test))
#     f1 = f1_score(y_test, y_pred, average='weighted' if n_classes > 2 else 'binary')
#
#     # AUC
#     try:
#         if y_proba is not None:
#             if n_classes == 2:
#                 auc = roc_auc_score(y_test, y_proba[:, 1])
#             else:
#                 auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
#         else:
#             auc = np.nan
#     except:
#         auc = np.nan
#
#     return {
#         'accuracy':  accuracy,
#         'f1': f1,
#         'roc_auc':  auc,
#         'train_time':  train_time
#     }
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    """评估单个模型"""
    start_time = time.time()

    # 训练
    if 'early_stop' in model_name and hasattr(model, 'fit'):
        # XGBoost早停需要eval_set
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    train_time = time.time() - start_time

    # 预测 (同时预测测试集和训练集)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None

    # 计算指标
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    # 论文核心指标：泛化差距 (Gap) = 训练集准确率 - 测试集准确率
    gap = train_accuracy - test_accuracy

    f1 = f1_score(y_test, y_pred_test, average='weighted')

    # AUC
    try:
        if y_proba is not None:
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        else:
            auc = np.nan
    except:
        auc = np.nan

    return {
        'accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'gap': gap,
        'f1': f1,
        'roc_auc': auc,
        'train_time': train_time
    }

def run_experiment(datasets=None, n_folds=5, random_state=42):
    """运行主实验"""
    print("=" * 70)
    print("Experiment 1: Main Comparison")
    print("=" * 70)
    
    # 获取数据集列表
    available_datasets = list_datasets()
    print(f"\nAvailable datasets: {len(available_datasets)}")
    
    if datasets is None: 
        # 排除合成数据集，只用真实数据集
        datasets = [d for d in available_datasets if not d.startswith('synthetic')]
    else:
        datasets = [d for d in datasets if d in available_datasets]
    
    if len(datasets) == 0:
        print("ERROR: No datasets found!")
        print(f"Looking in: {DATA_DIR}")
        print(f"Available:  {available_datasets}")
        return None, None
    
    print(f"Selected datasets: {len(datasets)}")
    print(f"  {datasets}")
    print(f"Folds: {n_folds}")
    
    # 获取模型
    baselines = get_baselines()
    adaprune_models = get_adaprune_models()
    all_models = {**baselines, **adaprune_models}
    
    print(f"\nModels: {len(all_models)}")
    print(f"  Baselines: {list(baselines.keys())}")
    print(f"  AdaPrune:  {list(adaprune_models.keys())}")
    
    # 存储结果
    results = []
    
    # 遍历数据集
    for dataset_name in tqdm(datasets, desc="Datasets"):
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            X, y, metadata = load_dataset(dataset_name)
            print(f"  Shape: X={X.shape}, y={y.shape}, classes={len(np.unique(y))}")
        except Exception as e: 
            print(f"  Error loading: {e}")
            continue
        
        # 交叉验证
        n_classes = len(np. unique(y))
        if n_classes < 2:
            print(f"  Skipping:  only {n_classes} class")
            continue
            
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        for fold, (train_idx, test_idx) in enumerate(kfold. split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # 评估每个模型
            for model_name, model_fn in all_models.items():
                try:
                    model = model_fn()  # 创建新模型实例
                    
                    metrics = evaluate_model(
                        model, X_train, y_train, X_test, y_test
                    )
                    
                    results.append({
                        'dataset': dataset_name,
                        'fold': fold,
                        'method': model_name,
                        **metrics
                    })
                    
                except Exception as e: 
                    print(f"    Error {model_name} fold {fold}: {e}")
                    continue
            
            # 打印第一折结果
            if fold == 0:
                fold_results = [r for r in results 
                               if r['dataset'] == dataset_name and r['fold'] == 0]
                print(f"\n  Fold 0 Results:")
                for r in sorted(fold_results, key=lambda x:  -x['accuracy']):
                    print(f"    {r['method']:20s}:  Acc={r['accuracy']:.4f}, F1={r['f1']:.4f}")
    
    if len(results) == 0:
        print("\nERROR: No results collected!")
        return None, None
    
    # 转换为DataFrame
    results_df = pd. DataFrame(results)
    
    # 保存原始结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(RESULTS_DIR / f'exp01_raw_results_{timestamp}.csv', index=False)
    
    # 生成汇总表
    summary = generate_summary(results_df)
    summary.to_csv(RESULTS_DIR / f'exp01_summary_{timestamp}.csv')
    
    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    print(f"\nResults saved to:  {RESULTS_DIR}")
    
    # 打印汇总
    print("\n" + "=" * 70)
    print("Summary (by Method)")
    print("=" * 70)
    print(summary. to_string())
    
    return results_df, summary


# def generate_summary(results_df):
#     """生成汇总表"""
#     # 按方法聚合
#     summary = results_df. groupby('method').agg({
#         'accuracy': ['mean', 'std'],
#         'f1': ['mean', 'std'],
#         'roc_auc': ['mean', 'std'],
#         'train_time':  'mean'
#     }).round(4)
#
#     # 展平列名
#     summary.columns = ['_'.join(col).strip() for col in summary.columns]
#
#     # 计算排名
#     summary['rank'] = summary['accuracy_mean'].rank(ascending=False)
#
#     # 排序
#     summary = summary.sort_values('accuracy_mean', ascending=False)
#
#     return summary
def generate_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """生成汇总表"""
    # 按方法聚合 (加上 gap 和 train_accuracy)
    summary = results_df.groupby('method').agg({
        'accuracy': ['mean', 'std'],
        'train_accuracy': ['mean', 'std'],
        'gap': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'roc_auc': ['mean', 'std'],
        'train_time': 'mean'
    }).round(4)

    # 展平列名
    summary.columns = ['_'.join(col).strip() for col in summary.columns]

    # 计算排名
    summary['rank'] = summary['accuracy_mean'].rank(ascending=False)

    # 排序
    summary = summary.sort_values('accuracy_mean', ascending=False)

    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=None, 
                        help='Specific datasets to use')
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser. parse_args()
    
    run_experiment(
        datasets=args.datasets,
        n_folds=args. folds,
        random_state=args.seed
    )