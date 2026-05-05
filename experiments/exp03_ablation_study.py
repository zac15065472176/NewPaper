"""
实验3: 消融实验
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.  model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 项目路径 - 使用 os.path 确保兼容性
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from adaprune import AdaPruneGBDT

# 目录设置
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

print(f"PROJECT_ROOT: {PROJECT_ROOT}")
print(f"DATA_DIR: {DATA_DIR}")


def load_dataset(name):
    """加载数据集"""
    file_path = DATA_DIR / f"{name}.pkl"
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['X'], data['y'], data. get('metadata', {})


def list_datasets():
    """列出所有数据集"""
    if not DATA_DIR. exists():
        print(f"  Directory does not exist: {DATA_DIR}")
        return []
    
    # 使用 os.listdir 作为备选
    try:
        files = [f[:-4] for f in os.listdir(DATA_DIR) if f.endswith('.pkl')]
        return files
    except Exception as e:
        print(f"  Error listing files: {e}")
        return []


def run_ablation_study(n_runs=3, random_state=42):
    """运行消融实验"""
    print("=" * 70)
    print("Experiment 3: Ablation Study")
    print("=" * 70)
    
    # 确保目录存在
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # 检查数据目录
    print(f"\nData directory:  {DATA_DIR}")
    print(f"Exists: {DATA_DIR.exists()}")
    
    # 列出目录内容
    if DATA_DIR.exists():
        try:
            files_in_dir = os. listdir(DATA_DIR)
            print(f"Files in directory: {len(files_in_dir)}")
            if len(files_in_dir) > 0:
                print(f"  First 5: {files_in_dir[:5]}")
        except Exception as e:
            print(f"Error listing:  {e}")
    
    # 获取数据集
    available_datasets = list_datasets()
    print(f"Available datasets: {len(available_datasets)}")
    
    if len(available_datasets) == 0:
        print("\nNo datasets found.  Generating test data...")
        # 生成临时测试数据
        from sklearn.datasets import make_classification
        
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        datasets = ['test_data']
        test_data = {'X': X. astype(np.float32), 'y': y. astype(np. int32)}
    else:
        # 选择数据集
        # target = ['diabetes', 'ionosphere', 'credit-g', 'spambase', 'vehicle']
        # datasets = [d for d in target if d in available_datasets]
        #
        # if len(datasets) == 0:
        #     datasets = available_datasets[: 5]
        # 使用全部 21 个数据集进行消融实验
        # datasets = available_datasets
        # 必须严格锁定为主实验中使用的 21 个真实数据集！
        target = [
            'adult', 'bank-marketing', 'credit-default', 'credit-g',
            'diabetes', 'dry-bean', 'eeg-eye-state', 'electricity',
            'ionosphere', 'kc1', 'letter', 'magic', 'mushroom',
            'phoneme', 'satimage', 'segment', 'sonar', 'spambase',
            'tic-tac-toe', 'vehicle', 'waveform'
        ]

        # 过滤掉合成数据集和其他多余数据
        datasets = [d for d in target if d in available_datasets]
        print(f"Filtered datasets for ablation ({len(datasets)}): {datasets}")

        test_data = None
    
    print(f"Using datasets: {datasets}")
    print(f"Runs: {n_runs}")
    
    # 消融配置
    ablation_configs = {
        'Full_hybrid': {'strategy': 'hybrid', 'adaptation_frequency': 5},
        'Gap_only': {'strategy': 'gap_based', 'adaptation_frequency': 5},
        'Trend_only': {'strategy': 'trend_based', 'adaptation_frequency': 5},
        'DataAware_only': {'strategy': 'data_aware', 'adaptation_frequency': 5},
        'No_adaptation': {'strategy': 'hybrid', 'adaptation_frequency': 10000},
    }
    
    print(f"Configs: {list(ablation_configs.keys())}")
    
    results = []
    
    for dataset_name in tqdm(datasets, desc="Datasets"):
        print(f"\n--- {dataset_name} ---")
        
        try:
            if test_data is not None:
                X, y = test_data['X'], test_data['y']
            else:
                X, y, _ = load_dataset(dataset_name)
            print(f"  Loaded:  X={X.shape}")
        except Exception as e: 
            print(f"  Error:  {e}")
            continue
        
        for run in range(n_runs):
            seed = random_state + run
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=seed, stratify=y
                )
            except: 
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=seed
                )
            
            for config_name, config in ablation_configs.items():
                try:
                    model = AdaPruneGBDT(
                        n_estimators=50,
                        random_state=seed,
                        verbose=0,
                        **config
                    )
                    
                    model.fit(X_train, y_train)
                    
                    train_acc = accuracy_score(y_train, model.predict(X_train))
                    test_acc = accuracy_score(y_test, model.predict(X_test))
                    
                    results.append({
                        'dataset': dataset_name,
                        'run': run,
                        'config': config_name,
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'gap': train_acc - test_acc,
                    })
                except Exception as e: 
                    print(f"    {config_name} error: {e}")
    
    if len(results) == 0:
        print("No results collected!")
        return None, None
    
    # 结果处理
    results_df = pd.DataFrame(results)
    
    summary = results_df. groupby('config').agg({
        'test_accuracy': ['mean', 'std'],
        'gap':  ['mean', 'std'],
    }).round(4)
    summary.columns = ['acc_mean', 'acc_std', 'gap_mean', 'gap_std']
    summary = summary.sort_values('acc_mean', ascending=False)
    
    # 保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(RESULTS_DIR / f'exp03_ablation_{timestamp}. csv', index=False)
    summary.to_csv(RESULTS_DIR / f'exp03_summary_{timestamp}.csv')
    
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(summary. to_string())
    
    print(f"\nSaved to: {RESULTS_DIR}")
    
    return results_df, summary


if __name__ == "__main__": 
    run_ablation_study(n_runs=3)