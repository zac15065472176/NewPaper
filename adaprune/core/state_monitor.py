"""
状态监控器

监控训练过程中的模型状态，检测过拟合/欠拟合
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class TrainingState: 
    """训练状态数据类"""
    iteration: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    train_metric: float = 0.0
    val_metric: float = 0.0
    gap: float = 0.0
    gap_trend: float = 0.0
    overfit_score: float = 0.0
    underfit_score: float = 0.0
    plateau_score: float = 0.0


class StateMonitor: 
    """
    状态监控器
    
    监控训练过程中的模型状态，计算关键指标：
    - 泛化差距 (gap)
    - 差距趋势 (gap_trend)
    - 过拟合分数 (overfit_score)
    - 欠拟合分数 (underfit_score)
    - 平台期分数 (plateau_score)
    """
    
    def __init__(self, window_size: int = 10):
        """
        初始化监控器
        
        Parameters
        ----------
        window_size : int
            计算趋势时的滑动窗口大小
        """
        self.window_size = window_size
        
        # 历史记录
        self. train_losses:  List[float] = []
        self. val_losses: List[float] = []
        self.train_metrics: List[float] = []
        self.val_metrics: List[float] = []
        self.complexities: List[float] = []
        
        # 当前状态
        self.current_state = TrainingState()
    
    def reset(self):
        """重置监控器"""
        self.train_losses = []
        self. val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.complexities = []
        self. current_state = TrainingState()
    
    def update(self,
               train_loss: float,
               val_loss: float,
               train_metric: float,
               val_metric: float,
               model_complexity: float = 0.0):
        """
        更新监控状态
        
        Parameters
        ----------
        train_loss :  float
            训练集损失
        val_loss : float
            验证集损失
        train_metric : float
            训练集指标（如准确率）
        val_metric : float
            验证集指标
        model_complexity :  float
            模型复杂度（如叶子节点数）
        """
        self. train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.train_metrics.append(train_metric)
        self.val_metrics.append(val_metric)
        self.complexities.append(model_complexity)
        
        self._compute_current_state()
    
    def _compute_current_state(self):
        """计算当前状态的各种指标"""
        n = len(self.train_metrics)
        
        if n < 2:
            self.current_state = TrainingState(
                iteration=n - 1,
                train_loss=self.train_losses[-1] if self.train_losses else 0,
                val_loss=self.val_losses[-1] if self.val_losses else 0,
                train_metric=self.train_metrics[-1] if self.train_metrics else 0,
                val_metric=self.val_metrics[-1] if self. val_metrics else 0,
            )
            return
        
        # 当前泛化差距
        current_gap = self. train_metrics[-1] - self.val_metrics[-1]
        
        # 差距趋势
        gap_trend = self._compute_gap_trend()
        
        # 验证集改进
        val_improvement = self._compute_val_improvement()
        
        # 过拟合分数
        overfit_score = self._compute_overfit_score(current_gap, gap_trend)
        
        # 欠拟合分数
        underfit_score = self._compute_underfit_score()
        
        # 平台期分数
        plateau_score = self._compute_plateau_score(val_improvement)
        
        self.current_state = TrainingState(
            iteration=n - 1,
            train_loss=self.train_losses[-1],
            val_loss=self.val_losses[-1],
            train_metric=self.train_metrics[-1],
            val_metric=self.val_metrics[-1],
            gap=current_gap,
            gap_trend=gap_trend,
            overfit_score=overfit_score,
            underfit_score=underfit_score,
            plateau_score=plateau_score,
        )
    
    def _compute_gap_trend(self) -> float:
        """计算差距趋势"""
        n = len(self. train_metrics)
        
        if n < self.window_size:
            if n < 2:
                return 0.0
            gaps = [self.train_metrics[i] - self.val_metrics[i] for i in range(n)]
        else:
            gaps = [self.train_metrics[i] - self. val_metrics[i] 
                   for i in range(-self.window_size, 0)]
        
        if len(gaps) < 2:
            return 0.0
        
        # 线性拟合斜率
        x = np.arange(len(gaps))
        try:
            slope = np.polyfit(x, gaps, 1)[0]
            return float(slope)
        except Exception:
            return 0.0
    
    def _compute_val_improvement(self) -> float:
        """计算验证集改进"""
        n = len(self.val_metrics)
        
        if n < self.window_size:
            if n < 2:
                return 0.0
            return self. val_metrics[-1] - self.val_metrics[0]
        
        recent_val = self.val_metrics[-self.window_size:]
        
        if n >= 2 * self.window_size:
            previous_val = self. val_metrics[-2*self.window_size:-self.window_size]
        else:
            previous_val = self. val_metrics[: self.window_size]
        
        return float(np.mean(recent_val) - np.mean(previous_val))
    
    def _compute_overfit_score(self, gap: float, trend: float) -> float:
        """
        计算过拟合分数 [0, 1]
        综合考虑当前差距和差距趋势
        """
        # 差距贡献（使用tanh映射）
        gap_score = np.tanh(gap * 5)
        gap_score = max(0, gap_score)
        
        # 趋势贡献
        trend_score = np.tanh(trend * 50)
        trend_score = max(0, trend_score)
        
        # 综合分数
        return float(0.6 * gap_score + 0.4 * trend_score)
    
    def _compute_underfit_score(self) -> float:
        """
        计算欠拟合分数 [0, 1]
        训练和验证性能都较差时分数高
        """
        if len(self.train_metrics) < 5:
            return 0.0
        
        train_level = self.train_metrics[-1]
        val_level = self.val_metrics[-1]
        
        # 如果两者都较低且接近，可能欠拟合
        if train_level < 0.7 and val_level < 0.7:
            gap = abs(train_level - val_level)
            if gap < 0.05: 
                return float(1 - (train_level + val_level) / 2)
        
        return 0.0
    
    def _compute_plateau_score(self, improvement: float) -> float:
        """
        计算平台期分数 [0, 1]
        验证集性能长期无改进时分数高
        """
        if len(self. val_metrics) < self.window_size:
            return 0.0
        
        # 改进很小时进入平台期
        plateau_score = 1 / (1 + np.exp(improvement * 100))
        return float(plateau_score)
    
    def get_state(self) -> Dict:
        """获取当前状态字典"""
        return {
            'iteration': self.current_state.iteration,
            'train_loss': self.current_state.train_loss,
            'val_loss': self. current_state.val_loss,
            'train_metric':  self.current_state.train_metric,
            'val_metric': self.current_state.val_metric,
            'gap': self.current_state.gap,
            'gap_trend': self.current_state.gap_trend,
            'overfit_score':  self.current_state.overfit_score,
            'underfit_score': self.current_state. underfit_score,
            'plateau_score': self.current_state.plateau_score,
        }
    
    def get_history(self) -> Dict:
        """获取完整历史"""
        return {
            'train_losses': self. train_losses. copy(),
            'val_losses': self.val_losses. copy(),
            'train_metrics': self.train_metrics. copy(),
            'val_metrics': self.val_metrics. copy(),
            'complexities': self. complexities.copy(),
        }
    
    def is_overfitting(self, threshold: float = 0.5) -> bool:
        """判断是否过拟合"""
        return self.current_state.overfit_score > threshold
    
    def is_underfitting(self, threshold: float = 0.5) -> bool:
        """判断是否欠拟合"""
        return self.current_state.underfit_score > threshold
    
    def is_plateau(self, threshold:  float = 0.5) -> bool:
        """判断是否进入平台期"""
        return self. current_state.plateau_score > threshold