"""
剪枝控制器

管理剪枝参数的自适应调整
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, List


@dataclass
class PruningParams:
    """剪枝参数数据类"""
    max_depth: int = 10
    min_samples_leaf: int = 1
    min_samples_split: int = 2
    max_features: Optional[float] = None
    min_impurity_decrease: float = 0.0
    
    # XGBoost/LightGBM 特有参数
    min_child_weight: float = 1.0
    gamma: float = 0.0
    reg_alpha: float = 0.0
    reg_lambda:  float = 1.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'min_samples_split': self.min_samples_split,
            'max_features': self.max_features,
            'min_impurity_decrease': self.min_impurity_decrease,
            'min_child_weight':  self.min_child_weight,
            'gamma': self. gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
        }
    
    def copy(self) -> 'PruningParams':
        """创建副本"""
        return PruningParams(**self.to_dict())
    
    def to_sklearn_params(self) -> Dict:
        """转换为sklearn决策树参数"""
        return {
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'min_samples_split': self. min_samples_split,
            'max_features': self. max_features,
            'min_impurity_decrease': self.min_impurity_decrease,
        }
    
    def to_xgboost_params(self) -> Dict:
        """转换为XGBoost参数"""
        return {
            'max_depth': self. max_depth,
            'min_child_weight': self. min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda':  self.reg_lambda,
        }


class AdaptationStrategy(ABC):
    """自适应策略的抽象基类"""
    
    @abstractmethod
    def adapt(self,
              current_params: PruningParams,
              state:  Dict,
              data_profile: Dict,
              iteration: int) -> PruningParams: 
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
        iteration : int
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


class GapBasedStrategy(AdaptationStrategy):
    """
    基于训练-验证差距的自适应策略
    
    核心思想：
    - 差距大且持续扩大 → 增强剪枝
    - 差距小且两者都在提升 → 保持或减弱剪枝
    """
    
    def __init__(self,
                 gap_threshold: float = 0.05,
                 trend_threshold: float = 0.001,
                 depth_step: int = 1,
                 samples_factor: float = 1.5):
        self.gap_threshold = gap_threshold
        self.trend_threshold = trend_threshold
        self.depth_step = depth_step
        self.samples_factor = samples_factor
    
    @property
    def name(self) -> str:
        return "gap_based"
    
    def adapt(self, current_params, state, data_profile, iteration):
        new_params = current_params.copy()
        
        gap = state. get('gap', 0)
        overfit_score = state.get('overfit_score', 0)
        underfit_score = state.get('underfit_score', 0)
        
        n_samples = data_profile. get('n_samples', 1000)
        
        # 过拟合检测
        if overfit_score > 0.5 and gap > self.gap_threshold:
            # 增强剪枝
            new_params.max_depth = max(3, new_params.max_depth - self.depth_step)
            new_params.min_samples_leaf = min(
                n_samples // 10,
                int(new_params. min_samples_leaf * self.samples_factor)
            )
            new_params.gamma = new_params.gamma + 0.1
            new_params.reg_lambda = new_params. reg_lambda * 1.2
        
        # 欠拟合检测
        elif underfit_score > 0.5:
            # 减弱剪枝
            new_params.max_depth = min(20, new_params. max_depth + self.depth_step)
            new_params.min_samples_leaf = max(1, new_params. min_samples_leaf // 2)
            new_params. gamma = max(0, new_params.gamma - 0.1)
        
        return new_params


class TrendBasedStrategy(AdaptationStrategy):
    """
    基于性能趋势的自适应策略
    
    核心思想：
    - 验证性能持续提升 → 保持当前参数
    - 验证性能停滞 → 增强剪枝
    - 验证性能下降 → 强力剪枝
    """
    
    def __init__(self,
                 improvement_threshold: float = 0.001,
                 plateau_patience: int = 5):
        self.improvement_threshold = improvement_threshold
        self. plateau_patience = plateau_patience
        self.plateau_counter = 0
    
    @property
    def name(self) -> str:
        return "trend_based"
    
    def adapt(self, current_params, state, data_profile, iteration):
        new_params = current_params.copy()
        
        val_improvement = state. get('val_improvement', 0)
        plateau_score = state.get('plateau_score', 0)
        
        # 性能提升中，保持参数
        if val_improvement > self.improvement_threshold:
            self.plateau_counter = 0
            return new_params
        
        # 进入平台期
        if plateau_score > 0.5:
            self.plateau_counter += 1
            
            if self.plateau_counter >= self. plateau_patience: 
                # 触发剪枝增强
                new_params.max_depth = max(3, new_params.max_depth - 1)
                new_params.min_samples_leaf = min(
                    50, int(new_params.min_samples_leaf * 1.3)
                )
                self.plateau_counter = 0
        
        return new_params


class DataAwareStrategy(AdaptationStrategy):
    """
    数据感知的自适应策略（核心创新）
    
    核心思想：
    - 根据数据特征调整适应的速度和强度
    - 高噪声数据：更快响应，更强剪枝
    - 小样本数据：更敏感的过拟合检测
    - 高维数据：调整特征采样
    """
    
    def __init__(self,
                 base_gap_threshold: float = 0.05,
                 base_depth_step: int = 1):
        self.base_gap_threshold = base_gap_threshold
        self.base_depth_step = base_depth_step
    
    @property
    def name(self) -> str:
        return "data_aware"
    
    def adapt(self, current_params, state, data_profile, iteration):
        new_params = current_params. copy()
        
        # 获取数据特征
        sensitivity = data_profile. get('adaptation_sensitivity', 1.0)
        noise_level = data_profile.get('estimated_noise_level', 0.1)
        n_samples = data_profile. get('n_samples', 1000)
        n_features = data_profile.get('n_features', 10)
        class_overlap = data_profile. get('class_overlap_score', 0.5)
        
        # 动态阈值
        gap_threshold = self. base_gap_threshold / max(sensitivity, 0.1)
        depth_step = max(1, int(self.base_depth_step * sensitivity))
        
        gap = state.get('gap', 0)
        overfit_score = state.get('overfit_score', 0)
        underfit_score = state. get('underfit_score', 0)
        
        # === 过拟合响应 ===
        if overfit_score > 0.3: 
            # 噪声越高，剪枝越强
            noise_factor = 1 + noise_level * 2
            
            # 样本越少，剪枝越强
            size_factor = np.sqrt(1000 / max(n_samples, 100))
            
            # 综合调整强度
            adjustment_strength = noise_factor * size_factor * overfit_score
            
            # 调整max_depth
            depth_reduction = int(depth_step * adjustment_strength)
            new_params. max_depth = max(3, new_params.max_depth - depth_reduction)
            
            # 调整min_samples_leaf
            samples_increase = 1 + 0.5 * adjustment_strength
            new_params.min_samples_leaf = min(
                n_samples // 20,
                max(1, int(new_params.min_samples_leaf * samples_increase))
            )
            
            # 调整正则化参数
            new_params.gamma += 0.05 * adjustment_strength
            new_params.reg_lambda *= (1 + 0.1 * adjustment_strength)
        
        # === 欠拟合响应 ===
        elif underfit_score > 0.5:
            if class_overlap > 0.5:
                new_params.max_depth = min(20, new_params. max_depth + 2)
            else:
                new_params.max_depth = min(15, new_params. max_depth + 1)
            
            new_params. min_samples_leaf = max(1, new_params.min_samples_leaf - 1)
        
        # === 特征采样调整 ===
        if n_features > 50:
            correlation = data_profile.get('feature_correlation_mean', 0.3)
            if correlation > 0.5:
                current_max_features = new_params.max_features or 1.0
                new_params.max_features = max(0.3, current_max_features - 0.1)
        
        return new_params


class HybridStrategy(AdaptationStrategy):
    """
    混合策略
    
    核心思想：
    - 训练早期：使用数据感知策略，快速找到合适的初始参数
    - 训练中期：使用Gap-Based策略，精细调整
    - 训练后期：使用Trend-Based策略，稳定收敛
    """
    
    def __init__(self, total_iterations: int = 100):
        self.total_iterations = total_iterations
        self.gap_strategy = GapBasedStrategy()
        self.trend_strategy = TrendBasedStrategy()
        self.data_aware_strategy = DataAwareStrategy()
    
    @property
    def name(self) -> str:
        return "hybrid"
    
    def adapt(self, current_params, state, data_profile, iteration):
        progress = iteration / max(self.total_iterations, 1)
        
        if progress < 0.3:
            # 早期：数据���知
            return self.data_aware_strategy.adapt(
                current_params, state, data_profile, iteration
            )
        elif progress < 0.7:
            # 中期：Gap-Based
            return self.gap_strategy.adapt(
                current_params, state, data_profile, iteration
            )
        else:
            # 后期：Trend-Based
            return self.trend_strategy.adapt(
                current_params, state, data_profile, iteration
            )


class PruningController: 
    """
    剪枝控制器
    
    管理自适应策略并提供统一接口
    """
    
    STRATEGIES = {
        'gap_based': GapBasedStrategy,
        'trend_based': TrendBasedStrategy,
        'data_aware':  DataAwareStrategy,
        'hybrid':  HybridStrategy,
    }
    
    def __init__(self,
                 strategy:  str = 'hybrid',
                 initial_params: Optional[PruningParams] = None,
                 adaptation_frequency: int = 5,
                 total_iterations: int = 100,
                 **strategy_kwargs):
        """
        初始化剪枝控制器
        
        Parameters
        ----------
        strategy :  str
            策略名称 ['gap_based', 'trend_based', 'data_aware', 'hybrid']
        initial_params : PruningParams, optional
            初始剪枝参数
        adaptation_frequency : int
            每隔多少轮调整一次参数
        total_iterations : int
            总迭代次数（用于Hybrid策略）
        """
        self.strategy_name = strategy
        self.adaptation_frequency = adaptation_frequency
        
        # 创建策略实例
        if strategy == 'hybrid':
            self. strategy = HybridStrategy(total_iterations=total_iterations)
        else:
            strategy_class = self. STRATEGIES.get(strategy, HybridStrategy)
            self.strategy = strategy_class(**strategy_kwargs)
        
        # 初始参数
        if initial_params is None:
            self.current_params = PruningParams()
        else:
            self.current_params = initial_params. copy()
        
        # 参数历史
        self. param_history:  List[Dict] = []
    
    def reset(self):
        """重置控制器"""
        self.current_params = PruningParams()
        self.param_history = []
    
    def initialize_from_data(self, data_profile: Dict):
        """根据数据特征初始化参数"""
        self.current_params. max_depth = data_profile.get(
            'recommended_initial_depth', 10
        )
        self.current_params.min_samples_leaf = data_profile.get(
            'recommended_min_samples', 1
        )
    
    def should_adapt(self, iteration: int) -> bool:
        """判断当前轮次是否需要调整参数"""
        return iteration > 0 and iteration % self.adaptation_frequency == 0
    
    def adapt(self,
              state: Dict,
              data_profile: Dict,
              iteration:  int) -> PruningParams:
        """
        执行参数自适应
        
        Parameters
        ----------
        state : dict
            当前模型状态
        data_profile : dict
            数据特征
        iteration :  int
            当前迭代轮次
        
        Returns
        -------
        params : PruningParams
            当前剪枝参数
        """
        if self.should_adapt(iteration):
            self.current_params = self. strategy.adapt(
                self.current_params, state, data_profile, iteration
            )
        
        # 记录历史
        self. param_history.append({
            'iteration': iteration,
            **self.current_params.to_dict()
        })
        
        return self. current_params
    
    def get_current_params(self) -> PruningParams:
        """获取当前参数"""
        return self.current_params. copy()
    
    def get_history(self) -> List[Dict]:
        """获取参数历史"""
        return self.param_history.copy()
    
    def get_sklearn_params(self) -> Dict:
        """获取sklearn格式参数"""
        return self.current_params. to_sklearn_params()
    
    def get_xgboost_params(self) -> Dict:
        """获取XGBoost格式参数"""
        return self.current_params.to_xgboost_params()