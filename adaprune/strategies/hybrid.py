"""
Hybrid 混合策略
"""

from typing import Dict
from .base import AdaptationStrategy
from . gap_based import GapBasedStrategy
from .trend_based import TrendBasedStrategy
from .data_aware import DataAwareStrategy
from .. core.pruning_controller import PruningParams


class HybridStrategy(AdaptationStrategy):
    """
    混合策略
    
    核心思想：
    - 训练早期：使用数据感知策略，快速找到合适的初始参数
    - 训练中期：使用Gap-Based策略，精细调整
    - 训练后期：使用Trend-Based策略，稳定收敛
    
    Parameters
    ----------
    total_iterations : int
        总迭代次数
    early_phase :  float
        早期阶段比例
    mid_phase :  float
        中期阶段结束比例
    """
    
    def __init__(
        self,
        total_iterations: int = 100,
        early_phase: float = 0.3,
        mid_phase: float = 0.7
    ):
        self.total_iterations = total_iterations
        self.early_phase = early_phase
        self.mid_phase = mid_phase
        
        self.gap_strategy = GapBasedStrategy()
        self.trend_strategy = TrendBasedStrategy()
        self.data_aware_strategy = DataAwareStrategy()
    
    @property
    def name(self) -> str:
        return "hybrid"
    
    def reset(self):
        """重置所有子策略"""
        self.trend_strategy.reset()
    
    def adapt(
        self,
        current_params:  PruningParams,
        state: Dict,
        data_profile: Dict,
        iteration: int
    ) -> PruningParams:
        progress = iteration / max(self.total_iterations, 1)
        
        if progress < self.early_phase:
            # 早期：数据感知策略
            return self.data_aware_strategy.adapt(
                current_params, state, data_profile, iteration
            )
        elif progress < self.mid_phase:
            # 中期：Gap-Based策略
            return self. gap_strategy.adapt(
                current_params, state, data_profile, iteration
            )
        else:
            # 后期：Trend-Based策略
            return self.trend_strategy.adapt(
                current_params, state, data_profile, iteration
            )
    
    def get_current_phase(self, iteration: int) -> str:
        """获取当前阶段名称"""
        progress = iteration / max(self.total_iterations, 1)
        
        if progress < self.early_phase:
            return "early (data_aware)"
        elif progress < self.mid_phase:
            return "mid (gap_based)"
        else:
            return "late (trend_based)"