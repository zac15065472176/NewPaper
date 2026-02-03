"""
数据画像分析器

分析数据集特征，为自适应策略提供先验信息
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')


class DataProfileAnalyzer:
    """
    数据画像分析器
    
    分析数据集的关键特征，包括：
    - 基础统计信息
    - 分布特征
    - 复杂度特征
    - 噪声估计
    
    这些特征用于指导自适应剪枝策略
    """
    
    def __init__(self, random_state: int = 42):
        """
        初始化分析器
        
        Parameters
        ----------
        random_state : int
            随机种子
        """
        self.random_state = random_state
        self.profile = {}
        self._is_analyzed = False
    
    def analyze(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        分析数据集特征
        
        Parameters
        ----------
        X : np. ndarray, shape (n_samples, n_features)
            特征矩阵
        y : np.ndarray, shape (n_samples,)
            目标变量
        
        Returns
        -------
        profile : dict
            数据特征字典
        """
        np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # 基础统计
        self.profile['n_samples'] = n_samples
        self.profile['n_features'] = n_features
        self.profile['samples_per_feature'] = n_samples / max(n_features, 1)
        
        # 目标变量特征
        self. profile['n_classes'] = len(np.unique(y))
        self.profile['class_imbalance_ratio'] = self._compute_imbalance_ratio(y)
        self.profile['target_entropy'] = self._compute_entropy(y)
        
        # 特征空间特征
        self. profile['feature_sparsity'] = np.mean(X == 0)
        self.profile['feature_correlation_mean'] = self._mean_correlation(X)
        self.profile['intrinsic_dimensionality'] = self._estimate_intrinsic_dim(X)
        
        # 分布特征
        self.profile['feature_skewness_mean'] = self._compute_skewness(X)
        self.profile['feature_kurtosis_mean'] = self._compute_kurtosis(X)
        self.profile['feature_variance_cv'] = self._compute_variance_cv(X)
        
        # 噪声估计
        self.profile['estimated_noise_level'] = self._estimate_noise_level(X, y)
        
        # 复杂度特征
        self.profile['class_overlap_score'] = self._compute_class_overlap(X, y)
        
        # 派生特征：推荐的初始参数
        self. profile['recommended_initial_depth'] = self._recommend_initial_depth()
        self.profile['recommended_min_samples'] = self._recommend_min_samples()
        self.profile['adaptation_sensitivity'] = self._compute_sensitivity()
        
        self._is_analyzed = True
        
        return self.profile
    
    def _compute_imbalance_ratio(self, y: np. ndarray) -> float:
        """计算类别不平衡比率"""
        unique, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            return 1.0
        return float(counts. max() / counts.min())
    
    def _compute_entropy(self, y: np. ndarray) -> float:
        """计算目标变量熵"""
        unique, counts = np. unique(y, return_counts=True)
        probs = counts / len(y)
        return float(-np.sum(probs * np.log2(probs + 1e-10)))
    
    def _mean_correlation(self, X: np.ndarray) -> float:
        """计算特征间平均相关性"""
        n_features = X. shape[1]
        
        if n_features < 2:
            return 0.0
        
        # 高维时随机采样特征
        if n_features > 100:
            idx = np.random. choice(n_features, 100, replace=False)
            X_sample = X[: , idx]
        else:
            X_sample = X
        
        try:
            # 移除常数列
            var = np.var(X_sample, axis=0)
            valid_cols = var > 1e-10
            if np.sum(valid_cols) < 2:
                return 0.0
            
            X_valid = X_sample[:, valid_cols]
            corr_matrix = np.corrcoef(X_valid.T)
            
            # 处理NaN
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
            
            # 取上三角，排除对角线
            upper_tri = np.triu(corr_matrix, k=1)
            non_zero = upper_tri[upper_tri != 0]
            
            if len(non_zero) == 0:
                return 0.0
            
            return float(np.mean(np.abs(non_zero)))
        except Exception:
            return 0.0
    
    def _estimate_intrinsic_dim(self, X: np.ndarray, threshold: float = 0.95) -> int:
        """使用PCA估计内在维度"""
        n_features = X. shape[1]
        n_samples = X. shape[0]
        
        if n_features < 2 or n_samples < 2:
            return n_features
        
        try:
            n_components = min(n_features, n_samples, 50)
            pca = PCA(n_components=n_components, random_state=self.random_state)
            pca. fit(X)
            
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            intrinsic_dim = int(np.argmax(cumsum >= threshold) + 1)
            
            return max(1, intrinsic_dim)
        except Exception: 
            return n_features
    
    def _compute_skewness(self, X: np.ndarray) -> float:
        """计算特征偏度均值"""
        try:
            from scipy.stats import skew
            skewness = skew(X, axis=0, nan_policy='omit')
            return float(np.nanmean(np.abs(skewness)))
        except Exception:
            return 0.0
    
    def _compute_kurtosis(self, X: np.ndarray) -> float:
        """计算特征峰度均值"""
        try: 
            from scipy.stats import kurtosis
            kurt = kurtosis(X, axis=0, nan_policy='omit')
            return float(np. nanmean(np.abs(kurt)))
        except Exception: 
            return 0.0
    
    def _compute_variance_cv(self, X: np. ndarray) -> float:
        """计算特征方差的变异系数"""
        variances = np.var(X, axis=0)
        variances = variances[variances > 1e-10]
        
        if len(variances) == 0:
            return 0.0
        
        mean_var = np. mean(variances)
        std_var = np. std(variances)
        
        if mean_var < 1e-10:
            return 0.0
        
        return float(std_var / mean_var)
    
    def _estimate_noise_level(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        使用1-NN估计噪声水平
        噪声样本的1-NN通常与其标签不同
        """
        n_samples = X.shape[0]
        
        # 小样本使用简化估计
        if n_samples < 50:
            return 0.1
        
        try:
            # 采样以加速
            if n_samples > 2000:
                idx = np.random. choice(n_samples, 2000, replace=False)
                X_sample, y_sample = X[idx], y[idx]
            else:
                X_sample, y_sample = X, y
            
            knn = KNeighborsClassifier(n_neighbors=1)
            scores = cross_val_score(knn, X_sample, y_sample, cv=3, scoring='accuracy')
            
            # 1-NN的错误率可以近似噪声水平
            noise_estimate = 1 - np.mean(scores)
            return float(np.clip(noise_estimate, 0, 1))
        except Exception:
            return 0.1
    
    def _compute_class_overlap(self, X: np.ndarray, y: np. ndarray) -> float:
        """
        计算类别重叠度（基于Fisher准则）
        """
        classes = np.unique(y)
        n_classes = len(classes)
        
        if n_classes < 2:
            return 0.0
        
        if n_classes == 2:
            return self._binary_overlap(X, y, classes)
        else:
            return self._multiclass_overlap(X, y, classes)
    
    def _binary_overlap(self, X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
        """二分类重叠度"""
        try:
            X_c0 = X[y == classes[0]]
            X_c1 = X[y == classes[1]]
            
            if len(X_c0) < 2 or len(X_c1) < 2:
                return 0.5
            
            mean_diff = np.mean(X_c0, axis=0) - np.mean(X_c1, axis=0)
            pooled_var = (np.var(X_c0, axis=0) + np.var(X_c1, axis=0)) / 2 + 1e-10
            
            fisher_ratio = np.mean(mean_diff**2 / pooled_var)
            overlap_score = 1 / (1 + fisher_ratio)
            
            return float(overlap_score)
        except Exception: 
            return 0.5
    
    def _multiclass_overlap(self, X:  np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
        """多分类重叠度"""
        overlaps = []
        
        for i, c1 in enumerate(classes):
            for c2 in classes[i+1:]: 
                mask = (y == c1) | (y == c2)
                if np.sum(mask) > 10: 
                    overlap = self._binary_overlap(X[mask], y[mask], np.array([c1, c2]))
                    overlaps. append(overlap)
        
        if len(overlaps) == 0:
            return 0.5
        
        return float(np.mean(overlaps))
    
    def _recommend_initial_depth(self) -> int:
        """根据数据特征推荐初始树深度"""
        n = self.profile. get('n_samples', 1000)
        p = self.profile. get('n_features', 10)
        noise = self.profile.get('estimated_noise_level', 0.1)
        
        # 基础深度
        base_depth = int(np.log2(n / 10 + 1))
        base_depth = max(3, min(15, base_depth))
        
        # 噪声调整
        noise_penalty = int(noise * 5)
        
        # 特征数调整
        feature_bonus = 1 if p > 50 else 0
        
        recommended = max(3, min(15, base_depth - noise_penalty + feature_bonus))
        return recommended
    
    def _recommend_min_samples(self) -> int:
        """根据数据特征推荐初始min_samples_leaf"""
        n = self.profile. get('n_samples', 1000)
        imbalance = self. profile.get('class_imbalance_ratio', 1.0)
        
        # 基础值：样本数的1%
        base = max(1, int(n * 0.01))
        
        # 不平衡调整
        if imbalance > 5:
            base = max(1, base // 2)
        
        return min(50, base)
    
    def _compute_sensitivity(self) -> float:
        """
        计算自适应敏感度
        高噪声、小样本数据需要更敏感的调整
        """
        noise = self.profile.get('estimated_noise_level', 0.1)
        n = self.profile.get('n_samples', 1000)
        
        # 小样本敏感度高
        size_factor = 1.0 / np.log10(n + 10)
        # 高噪声敏感度高
        noise_factor = 1 + noise * 2
        
        return float(size_factor * noise_factor)
    
    def get_profile(self) -> Dict:
        """获取数据画像"""
        if not self._is_analyzed:
            raise ValueError("Please call analyze() first")
        return self.profile. copy()
    
    def get_summary(self) -> str:
        """获取数据画像摘要"""
        if not self._is_analyzed:
            return "Data not analyzed yet"
        
        summary = [
            "=" * 50,
            "Data Profile Summary",
            "=" * 50,
            f"Samples: {self. profile['n_samples']}",
            f"Features: {self.profile['n_features']}",
            f"Classes:  {self.profile['n_classes']}",
            f"Imbalance Ratio: {self. profile['class_imbalance_ratio']:.2f}",
            f"Estimated Noise:  {self.profile['estimated_noise_level']:.3f}",
            f"Class Overlap: {self.profile['class_overlap_score']:. 3f}",
            f"Intrinsic Dim: {self.profile['intrinsic_dimensionality']}",
            "-" * 50,
            f"Recommended max_depth: {self. profile['recommended_initial_depth']}",
            f"Recommended min_samples_leaf: {self. profile['recommended_min_samples']}",
            f"Adaptation Sensitivity: {self. profile['adaptation_sensitivity']:.3f}",
            "=" * 50,
        ]
        
        return "\n".join(summary)