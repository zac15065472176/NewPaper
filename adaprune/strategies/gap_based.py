"""
Gap-Based 自适应策略
"""

from typing import Dict
from . base import AdaptationStrategy
from ..core.pruning_controller import PruningParams


class GapBasedStrategy(AdaptationStrategy):
    """
    基于训练-验证差距的自适应策略
    
    核心思想：
    - 差距大且持续扩大 → 增强剪枝
    - 差距小且两者都在提升 → 保持或减弱剪枝
    
    Parameters
    ----------
    gap_threshold : float
        触发剪枝增强的差距阈值
    trend_threshold : float
        差距趋势阈值
    depth_step : int
        深度调整步长
    samples_factor : float
        min_samples_leaf调整因子
    """
    
    def __init__(
        self,
        gap_threshold: float = 0.05,
        trend_threshold: float = 0.001,
        depth_step: int = 1,
        samples_factor: float = 1.5
    ):
        self.gap_threshold = gap_threshold
        self.trend_threshold = trend_threshold
        self.depth_step = depth_step
        self. samples_factor = samples_factor
    
    @property
    def name(self) -> str:
        return "gap_based"
    
    def adapt(
        self,
        current_params:  PruningParams,
        state: Dict,
        data_profile: Dict,
        iteration: int
    ) -> PruningParams:
        new_params = current_params.copy()
        
        gap = state.get('gap', 0)
        overfit_score = state.get('overfit_score', 0)
        underfit_score = state.get('underfit_score', 0)
        
        n_samples = data_profile. get('n_samples', 1000)
        
        # 过拟合检测：增强剪枝
        if overfit_score > 0.5 and gap > self. gap_threshold: 
            new_params. max_depth = max(3, new_params.max_depth - self.depth_step)
            new_params.min_samples_leaf = min(
                n_samples // 10,
                int(new_params. min_samples_leaf * self.samples_factor)
            )
            new_params.gamma = new_params.gamma + 0.1
            new_params.reg_lambda = new_params.reg_lambda * 1.2
        
        # 欠拟合检测：减弱剪枝
        elif underfit_score > 0.5:
            new_params.max_depth = min(20, new_params. max_depth + self.depth_step)
            new_params.min_samples_leaf = max(1, new_params. min_samples_leaf // 2)
            new_params. gamma = max(0, new_params.gamma - 0.1)
        
        return new_params