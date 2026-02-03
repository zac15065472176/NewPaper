"""
自适应策略基类
"""

from abc import ABC, abstractmethod
from typing import Dict
from .. core.pruning_controller import PruningParams


class AdaptationStrategy(ABC):
    """自适应策略的抽象基类"""
    
    @abstractmethod
    def adapt(
        self,
        current_params: PruningParams,
        state:  Dict,
        data_profile: Dict,
        iteration: int
    ) -> PruningParams: 
        """
        根据当前状态调整剪枝参数
        
        Parameters
        ----------
        current_params : PruningParams
            当前剪枝参数
        state : dict
            当前模型状态
        data_profile : dict
            数据特征
        iteration :  int
            当前迭代轮次
        
        Returns
        -------
        new_params : PruningParams
            调整后的剪枝参数
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """策略名称"""
        pass
    
    def reset(self):
        """重置策略状态"""
        pass