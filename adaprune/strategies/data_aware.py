"""
Data-Aware 自适应策略（核心创新）
"""

import numpy as np
from typing import Dict
from .base import AdaptationStrategy
from ..core.pruning_controller import PruningParams


class DataAwareStrategy(AdaptationStrategy):
    """
    数据感知的自适应策略
    
    核心创新：
    - 根据数据特征调整适应的速度和强度
    - 高噪声数据：更快响应，更强剪枝
    - 小样本数据：更敏感的过拟合检测
    - 高维数据：调整特征采样
    
    Parameters
    ----------
    base_gap_threshold : float
        基础差距阈值
    base_depth_step : int
        基础深度调整步长
    """
    
    def __init__(
        self,
        base_gap_threshold:  float = 0.05,
        base_depth_step:  int = 1
    ):
        self.base_gap_threshold = base_gap_threshold
        self.base_depth_step = base_depth_step
    
    @property
    def name(self) -> str:
        return "data_aware"
    
    def adapt(
        self,
        current_params: PruningParams,
        state: Dict,
        data_profile: Dict,
        iteration: int
    ) -> PruningParams:
        new_params = current_params.copy()
        
        # 获取数据特征
        sensitivity = data_profile. get('adaptation_sensitivity', 1.0)
        noise_level = data_profile.get('estimated_noise_level', 0.1)
        n_samples = data_profile.get('n_samples', 1000)
        n_features = data_profile.get('n_features', 10)
        class_overlap = data_profile. get('class_overlap_score', 0.5)
        
        # 动态阈值和步长
        gap_threshold = self. base_gap_threshold / max(sensitivity, 0.1)
        depth_step = max(1, int(self.base_depth_step * sensitivity))
        
        gap = state.get('gap', 0)
        overfit_score = state.get('overfit_score', 0)
        underfit_score = state. get('underfit_score', 0)
        
        # === 过拟合响应 ===
        if overfit_score > 0.3: 
            # 噪声因子
            noise_factor = 1 + noise_level * 2
            # 样本量因子
            size_factor = np.sqrt(1000 / max(n_samples, 100))
            # 综合调整强度
            adjustment_strength = noise_factor * size_factor * overfit_score
            
            # 调整深度
            depth_reduction = int(depth_step * adjustment_strength)
            new_params.max_depth = max(3, new_params. max_depth - depth_reduction)
            
            # 调整min_samples_leaf
            samples_increase = 1 + 0.5 * adjustment_strength
            new_params.min_samples_leaf = min(
                n_samples // 20,
                max(1, int(new_params.min_samples_leaf * samples_increase))
            )
            
            # 调整正则化
            new_params.gamma += 0.05 * adjustment_strength
            new_params.reg_lambda *= (1 + 0.1 * adjustment_strength)
        
        # === 欠拟合响应 ===
        elif underfit_score > 0.5:
            if class_overlap > 0.5:
                new_params.max_depth = min(20, new_params. max_depth + 2)
            else:
                new_params.max_depth = min(15, new_params. max_depth + 1)
            
            new_params. min_samples_leaf = max(1, new_params.min_samples_leaf - 1)
        
        # === 高维数据��征采样 ===
        if n_features > 50:
            correlation = data_profile.get('feature_correlation_mean', 0.3)
            if correlation > 0.5:
                current_max_features = new_params.max_features or 1.0
                new_params.max_features = max(0.3, current_max_features - 0.1)
        
        return new_params