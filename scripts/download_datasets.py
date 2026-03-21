"""
数据集下载和预处理脚本
运行方式:  python scripts/download_datasets.py
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 数据存储目录
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'

# 创建目录
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ================= 核心数据集 (22个) =================
CORE_DATASETS = {
    'adult': {'openml_id': 1590, 'description': '人口收入预测'},
    'bank-marketing': {'openml_id': 1461, 'description': '银行营销预测'},
    'credit-g': {'openml_id': 31, 'description': '德国信用评估'},
    'diabetes': {'openml_id': 37, 'description': 'Pima糖尿病'},
    'ionosphere': {'openml_id': 59, 'description': '电离层雷达'},
    'sonar': {'openml_id': 40, 'description': '声纳信号'},
    'spambase': {'openml_id': 44, 'description': '垃圾邮件'},
    'vehicle': {'openml_id': 54, 'description': '车辆分类'},
    'segment': {'openml_id': 36, 'description': '图像分割'},
    'satimage': {'openml_id': 182, 'description': '卫星图像'},
    'electricity': {'openml_id': 151, 'description': '电价预测'},
    'phoneme': {'openml_id': 1489, 'description': '音素识别'},
    'magic': {'openml_id': 1120, 'description': 'MAGIC望远镜'},
    'eeg-eye-state': {'openml_id': 1471, 'description': 'EEG眼睛状态'},
    'waveform': {'openml_id': 60, 'description': '波形分类'},
    'amazon': {'openml_id': 1457, 'description': '亚马逊员工访问'},
    'dry-bean': {'openml_id': 42585, 'description': '干豆分类'},
    'letter': {'openml_id': 6, 'description': '英文字母识别'},
    'credit-default': {'openml_id': 42477, 'description': '信用卡违约预测'},
    'mushroom': {'openml_id': 24, 'description': '蘑菇毒性预测'},
    'tic-tac-toe': {'openml_id': 50, 'description': '井字棋胜负'},
    'kc1': {'openml_id': 1067, 'description': '软件缺陷预测'},
}

SYNTHETIC_CONFIGS = {
    'clean_large': {
        'n_samples': 10000, 'n_features': 20, 'n_informative': 15,
        'noise': 0.0, 'flip_y': 0.0, 'description': '干净大样本'
    },
    'clean_small': {
        'n_samples': 500, 'n_features': 20, 'n_informative': 15,
        'noise': 0.0, 'flip_y': 0.0, 'description': '干净小样本'
    },
    'noisy_large': {
        'n_samples': 10000, 'n_features': 20, 'n_informative': 15,
        'noise': 0.3, 'flip_y': 0.1, 'description': '噪声大样本'
    },
    'noisy_small': {
        'n_samples': 500, 'n_features': 20, 'n_informative': 15,
        'noise': 0.3, 'flip_y': 0.15, 'description': '噪声小样本'
    },
    'high_dim': {
        'n_samples': 1000, 'n_features': 200, 'n_informative': 20,
        'noise': 0.1, 'flip_y': 0.05, 'description': '高维数据'
    },
    'imbalanced': {
        'n_samples': 5000, 'n_features': 20, 'n_informative': 15,
        'noise': 0.1, 'flip_y': 0.05, 'weights': [0.9, 0.1],
        'description': '不平衡数据'
    },
}


def download_openml_dataset(dataset_id: int, dataset_name: str) -> Tuple:
    """从OpenML下载数据集"""
    try:
        import openml
        print(f"  Downloading {dataset_name} (ID: {dataset_id})...")

        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical, feature_names = dataset.get_data(
            target=dataset.default_target_attribute
        )

        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values

        return X, y, categorical, feature_names

    except Exception as e:
        print(f"  Error: {e}")
        return None, None, None, None


def preprocess_dataset(
        X: np.ndarray,
        y: np.ndarray,
        categorical: Optional[List[bool]] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """数据预处理"""
    X = X.copy()
    y = y.copy()

    # 处理缺失值
    for i in range(X.shape[1]):
        col = X[:, i]
        mask = pd.isna(col)
        if mask.any():
            if categorical and i < len(categorical) and categorical[i]:
                mode_val = pd.Series(col).mode()
                fill_val = mode_val.iloc[0] if len(mode_val) > 0 else 0
            else:
                try:
                    fill_val = np.nanmedian(col.astype(float))
                except:
                    fill_val = 0
            X[mask, i] = fill_val

    # 编码分类特征
    if categorical:
        for i, is_cat in enumerate(categorical):
            if is_cat and i < X.shape[1]:
                le = LabelEncoder()
                X[:, i] = le.fit_transform(X[:, i].astype(str))

    # 转换类型
    X = X.astype(np.float32)

    # 编码目标变量
    if y.dtype == object or (len(y) > 0 and isinstance(y[0], str)):
        le = LabelEncoder()
        y = le.fit_transform(y.astype(str))
    y = y.astype(np.int32)

    # 移除无效样本
    mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    X = X[mask]
    y = y[mask]

    return X, y


def generate_synthetic_dataset(config: dict, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """生成合成数据集"""
    params = {
        'n_samples': config.get('n_samples', 1000),
        'n_features': config.get('n_features', 20),
        'n_informative': config.get('n_informative', 10),
        'n_redundant': config.get('n_redundant', 5),
        'n_clusters_per_class': 2,
        'flip_y': config.get('flip_y', 0.0),
        'random_state': random_state,
    }

    if 'weights' in config:
        params['weights'] = config['weights']

    X, y = make_classification(**params)

    noise_level = config.get('noise', 0.0)
    if noise_level > 0:
        np.random.seed(random_state)
        X = X + np.random.randn(*X.shape) * noise_level

    return X.astype(np.float32), y.astype(np.int32)


def save_dataset(X: np.ndarray, y: np.ndarray, name: str, metadata: dict = None):
    """保存数据集"""
    save_path = PROCESSED_DIR / f"{name}.pkl"

    data = {
        'X': X,
        'y': y,
        'metadata': metadata or {}
    }

    with open(save_path, 'wb') as f:
        pickle.dump(data, f)

    print(f"  Saved: {name} (X: {X.shape}, y: {y.shape}, classes: {len(np.unique(y))})")


def download_all_datasets():
    """下载所有数据集"""
    print("=" * 60)
    print("AdaPrune Dataset Download Script")
    print("=" * 60)

    # 1. 下载OpenML数据集
    print("\n[1/2] Downloading OpenML Datasets...")
    print("-" * 40)

    success_count = 0
    for name, config in tqdm(CORE_DATASETS.items(), desc="OpenML"):
        save_path = PROCESSED_DIR / f"{name}.pkl"

        if save_path.exists():
            print(f"  {name}: Already exists, skipping...")
            success_count += 1
            continue

        X, y, cat_indicator, feature_names = download_openml_dataset(
            config['openml_id'], name
        )

        if X is not None:
            X, y = preprocess_dataset(X, y, cat_indicator)
            metadata = {
                'source': 'openml',
                'openml_id': config['openml_id'],
                'description': config.get('description', ''),
                'feature_names': feature_names,
            }
            save_dataset(X, y, name, metadata)
            success_count += 1

    print(f"\nOpenML datasets: {success_count}/{len(CORE_DATASETS)} successful")

    # 2. 生成合成数据集
    print("\n[2/2] Generating Synthetic Datasets...")
    print("-" * 40)

    for name, config in tqdm(SYNTHETIC_CONFIGS.items(), desc="Synthetic"):
        full_name = f"synthetic_{name}"
        save_path = PROCESSED_DIR / f"{full_name}.pkl"

        if save_path.exists():
            print(f"  {full_name}: Already exists, skipping...")
            continue

        X, y = generate_synthetic_dataset(config)
        metadata = {
            'source': 'synthetic',
            'config': config,
            'description': config.get('description', ''),
        }
        save_dataset(X, y, full_name, metadata)

    # 完成统计
    print("\n" + "=" * 60)
    print("Download Complete!")
    print("=" * 60)

    all_files = list(PROCESSED_DIR.glob("*.pkl"))
    print(f"\nTotal datasets: {len(all_files)}")
    print(f"Location: {PROCESSED_DIR}")

    print("\nDataset Summary:")
    print("-" * 60)
    print(f"{'Name':<25} {'Samples':>8} {'Features':>10} {'Classes':>8}")
    print("-" * 60)

    for f in sorted(all_files):
        with open(f, 'rb') as fp:
            data = pickle.load(fp)
        X, y = data['X'], data['y']
        print(f"{f.stem:<25} {X.shape[0]:>8} {X.shape[1]:>10} {len(np.unique(y)):>8}")


def verify_datasets():
    """验证数据集完整性"""
    print("\nVerifying datasets...")
    print("-" * 60)

    all_files = list(PROCESSED_DIR.glob("*.pkl"))
    issues = []

    for f in all_files:
        try:
            with open(f, 'rb') as fp:
                data = pickle.load(fp)
            X, y = data['X'], data['y']

            assert X.shape[0] == y.shape[0], "X and y length mismatch"
            assert not np.isnan(X).any(), "X contains NaN"
            assert not np.isinf(X).any(), "X contains Inf"
            assert len(np.unique(y)) >= 2, "Less than 2 classes"

            print(f"  ✓ {f.stem}")
        except Exception as e:
            print(f"  ✗ {f.stem}: {e}")
            issues.append((f.stem, str(e)))

    if issues:
        print(f"\n⚠ Found {len(issues)} issues")
    else:
        print(f"\n✓ All {len(all_files)} datasets verified!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download datasets for AdaPrune')
    parser.add_argument('--verify', action='store_true', help='Verify datasets only')
    parser.add_argument('--list', action='store_true', help='List available datasets')

    args = parser.parse_args()

    if args.verify:
        verify_datasets()
    elif args.list:
        print("Available datasets:")
        for f in sorted(PROCESSED_DIR.glob("*.pkl")):
            print(f"  - {f.stem}")
    else:
        download_all_datasets()
        verify_datasets()