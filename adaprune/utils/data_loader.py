"""
数据加载工具
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn. model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """
    数据加载器
    
    支持从OpenML下载数据集，或加载本地处理好的数据集
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化数据加载器
        
        Parameters
        ----------
        data_dir : str, optional
            数据目录路径
        """
        if data_dir is None:
            self. data_dir = Path(__file__).parent.parent. parent / 'data' / 'processed'
        else: 
            self.data_dir = Path(data_dir)
        
        self. data_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self, name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        加载数据集
        
        Parameters
        ----------
        name : str
            数据集名称
        
        Returns
        -------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            目标变量
        metadata : dict
            元数据
        """
        file_path = self. data_dir / f"{name}.pkl"
        
        if not file_path. exists():
            raise FileNotFoundError(
                f"Dataset '{name}' not found. "
                f"Run 'python scripts/download_datasets.py' first."
            )
        
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        return data['X'], data['y'], data. get('metadata', {})
    
    def save(
        self,
        X: np.ndarray,
        y: np.ndarray,
        name: str,
        metadata: Optional[Dict] = None
    ):
        """
        保存数据集
        
        Parameters
        ----------
        X : np.ndarray
            特征矩阵
        y : np. ndarray
            目标变量
        name : str
            数据集名称
        metadata : dict, optional
            元数据
        """
        file_path = self.data_dir / f"{name}.pkl"
        
        data = {
            'X': X,
            'y': y,
            'metadata': metadata or {}
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    
    def list_datasets(self) -> List[str]:
        """
        列出所有可用数据集
        
        Returns
        -------
        datasets : list
            数据集名称列表
        """
        return [f.stem for f in self. data_dir.glob("*. pkl")]
    
    def download_openml(
        self,
        dataset_id: int,
        name: str,
        force: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从OpenML下载数据集
        
        Parameters
        ----------
        dataset_id : int
            OpenML数据集ID
        name : str
            保存名称
        force :  bool
            是否强制重新下载
        
        Returns
        -------
        X : np.ndarray
            特征矩阵
        y : np.ndarray
            目标变量
        """
        import openml
        
        file_path = self. data_dir / f"{name}.pkl"
        
        if file_path.exists() and not force: 
            return self.load(name)[: 2]
        
        print(f"Downloading {name} from OpenML (ID: {dataset_id})...")
        
        dataset = openml.datasets.get_dataset(dataset_id)
        X, y, categorical, feature_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
        
        # 预处理
        X, y = self._preprocess(X, y, categorical)
        
        # 保存
        metadata = {
            'source': 'openml',
            'openml_id': dataset_id,
            'feature_names': feature_names,
        }
        self.save(X, y, name, metadata)
        
        return X, y
    
    def _preprocess(
        self,
        X: np.ndarray,
        y:  np.ndarray,
        categorical: Optional[List[bool]] = None
    ) -> Tuple[np.ndarray, np.ndarray]: 
        """预处理数据"""
        # 转换为numpy数组
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y. values
        
        # 处理缺失值
        X = np.nan_to_num(X, nan=0.0)
        
        # 编码分类特征
        if categorical: 
            for i, is_cat in enumerate(categorical):
                if is_cat: 
                    le = LabelEncoder()
                    X[: , i] = le.fit_transform(X[:, i]. astype(str))
        
        # 确保类型正确
        X = X.astype(np.float32)
        
        # 编码目标变量
        if y. dtype == object or isinstance(y[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y. astype(str))
        y = y.astype(np.int32)
        
        return X, y
    
    def generate_synthetic(
        self,
        name: str,
        n_samples: int = 1000,
        n_features: int = 20,
        n_informative: int = 10,
        n_redundant: int = 5,
        noise: float = 0.0,
        flip_y: float = 0.0,
        weights: Optional[List[float]] = None,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成合成数据集
        
        Parameters
        ----------
        name : str
            数据集名称
        n_samples : int
            样本数
        n_features :  int
            特征数
        n_informative : int
            有信息特征数
        n_redundant : int
            冗余特征数
        noise : float
            噪声水平
        flip_y : float
            标签翻转比例
        weights :  list, optional
            类别权重（用于不平衡数据）
        random_state : int
            随机种子
        
        Returns
        -------
        X : np.ndarray
            特征矩阵
        y : np. ndarray
            目标变量
        """
        params = {
            'n_samples': n_samples,
            'n_features':  n_features,
            'n_informative': n_informative,
            'n_redundant':  n_redundant,
            'n_clusters_per_class':  2,
            'flip_y': flip_y,
            'random_state': random_state,
        }
        
        if weights:
            params['weights'] = weights
        
        X, y = make_classification(**params)
        
        # 添加噪声
        if noise > 0:
            X = X + np.random.randn(*X.shape) * noise
        
        X = X.astype(np.float32)
        y = y.astype(np.int32)
        
        # 保存
        metadata = {
            'source': 'synthetic',
            'params': params,
            'noise': noise,
        }
        self. save(X, y, name, metadata)
        
        return X, y


# 便捷函数
_default_loader = None


def _get_loader() -> DataLoader: 
    global _default_loader
    if _default_loader is None: 
        _default_loader = DataLoader()
    return _default_loader


def load_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """加载数据集"""
    return _get_loader().load(name)


def list_datasets() -> List[str]:
    """列出可用数据集"""
    return _get_loader().list_datasets()