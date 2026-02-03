"""
Trend-Based 自适应策略
"""

from typing import Dict
from .base import AdaptationStrategy
from ..core. pruning_controller import PruningParams


class TrendBasedStrategy(AdaptationStrategy):
    """
    基于性能趋势的自适应策略
    
    核心思想：
    - 验证性能持续提升 → 保持当前参数
    - 验证性能停滞 → 增强剪枝
    - 验证性能下降 → 强力剪枝
    
    Parameters
    ----------
    improvement_threshold : float
        最小改进阈值
    plateau_patience : int
        平台期耐心值
    """
    
    def __init__(
        self,
        improvement_threshold: float = 0.001,
        plateau_patience: int = 5
    ):
        self.improvement_threshold = improvement_threshold
        self.plateau_patience = plateau_patience
        self.plateau_counter = 0
    
    @property
    def name(self) -> str:
        return "trend_based"
    
    def reset(self):
        """重置状态"""
        self.plateau_counter = 0
    
    def adapt(
        self,
        current_params: PruningParams,
        state: Dict,
        data_profile: Dict,
        iteration: int
    ) -> PruningParams:
        new_params = current_params.copy()
        
        val_improvement = state.get('val_improvement', 0)
        plateau_score = state.get('plateau_score', 0)
        
        # 性能提升中，保持参数
        if val_improvement > self. improvement_threshold: 
            self.plateau_counter = 0
            return new_params
        
        # 进入平台期
        if plateau_score > 0.5: 
            self.plateau_counter += 1
            
            if self.plateau_counter >= self. plateau_patience: 
                # 触发剪枝增强
                new_params.max_depth = max(3, new_params.max_depth - 1)
                new_params.min_samples_leaf = min(
                    50, int(new_params. min_samples_leaf * 1.3)
                )
                self.plateau_counter = 0
        
        return new_params