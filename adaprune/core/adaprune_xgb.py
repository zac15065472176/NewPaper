"""
AdaPrune XGBoost 优化版本 V2

改进：更平衡的准确率和过拟合控制
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import warnings

from .data_analyzer import DataProfileAnalyzer
from .state_monitor import StateMonitor

warnings.filterwarnings('ignore')


class AdaPruneXGB: 
    """
    AdaPrune XGBoost 优化版本 V2
    
    核心改进：
    1. 更平衡的初始参数（不过于保守）
    2. 渐进式参数调整（避免剧烈变化）
    3. 双向调整（可以增加也可以减少复杂度）
    4. 更智能的早停策略
    """
    
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.1,
        initial_max_depth: int = 6,
        max_depth_range: Tuple[int, int] = (3, 10),
        min_child_weight_range: Tuple[float, float] = (1, 15),
        adaptation_frequency: int = 15,
        gap_threshold: float = 0.03,
        early_stopping_rounds: int = 40,
        validation_fraction: float = 0.15,
        warmup_rounds: int = 30,
        random_state: Optional[int] = None,
        verbose: int = 0
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.initial_max_depth = initial_max_depth
        self. max_depth_range = max_depth_range
        self.min_child_weight_range = min_child_weight_range
        self.adaptation_frequency = adaptation_frequency
        self.gap_threshold = gap_threshold
        self. early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.warmup_rounds = warmup_rounds  # 预热期不调整参数
        self.random_state = random_state
        self. verbose = verbose
        
        # 组件
        self.data_analyzer = DataProfileAnalyzer(random_state=random_state)
        self.state_monitor = StateMonitor(window_size=10)
        
        # 模型状态
        self.model: Optional[xgb. Booster] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.label_encoder: Optional[LabelEncoder] = None
        self.data_profile: Dict = {}
        self.is_fitted: bool = False
        
        # 当前参数
        self.current_max_depth: int = initial_max_depth
        self.current_min_child_weight: float = 1.0
        self.current_gamma: float = 0.0
        self.current_reg_lambda: float = 1.0
        self.current_reg_alpha: float = 0.0
        self.current_subsample: float = 1.0
        self.current_colsample: float = 1.0
        
        # 历史记录
        self.param_history: List[Dict] = []
        self.train_scores: List[float] = []
        self.val_scores: List[float] = []
        self.best_iteration: int = 0
        self.best_val_score: float = 0.0
        self.best_model: Optional[xgb. Booster] = None
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'AdaPruneXGB': 
        """训练模型"""
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        
        # 标签编码
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder. classes_
        self.n_classes_ = len(self.classes_)
        
        # 划分验证集
        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction,
                random_state=self. random_state, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = self.label_encoder.transform(y_val)
        
        # 分析数据特征
        if self.verbose > 0:
            print("=" * 60)
            print("AdaPrune XGB V2 Training")
            print("=" * 60)
            print("\nAnalyzing data...")
        
        self.data_profile = self.data_analyzer.analyze(X_train, y_train)
        
        # 根据数据特征初始化参数
        self._initialize_params_from_data()
        
        if self.verbose > 0:
            print(f"  Samples: {X_train.shape[0]}, Features: {X_train.shape[1]}")
            print(f"  Noise level: {self.data_profile. get('estimated_noise_level', 0):.3f}")
            print(f"  Initial max_depth: {self.current_max_depth}")
            print(f"  Warmup rounds: {self.warmup_rounds}")
            print("\nTraining...")
            print("-" * 60)
        
        # 创建 DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # 重置状态
        self.state_monitor.reset()
        self.param_history = []
        self.train_scores = []
        self.val_scores = []
        
        best_val_score = 0.0
        best_iteration = 0
        rounds_without_improvement = 0
        
        # 增量训练
        self. model = None
        self.best_model = None
        trees_per_round = self.adaptation_frequency
        total_trees = 0
        
        while total_trees < self.n_estimators:
            # 获取当前 XGBoost 参数
            params = self._get_xgb_params()
            
            # 训练一批树
            if self.model is None:
                self.model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=trees_per_round,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    verbose_eval=False
                )
            else:
                self.model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=trees_per_round,
                    xgb_model=self.model,
                    evals=[(dtrain, 'train'), (dval, 'val')],
                    verbose_eval=False
                )
            
            total_trees += trees_per_round
            
            # 评估性能
            train_pred = self.model.predict(dtrain)
            val_pred = self.model.predict(dval)
            
            if self.n_classes_ == 2:
                train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
                val_acc = accuracy_score(y_val, (val_pred > 0.5).astype(int))
            else:
                train_acc = accuracy_score(y_train, train_pred. argmax(axis=1))
                val_acc = accuracy_score(y_val, val_pred.argmax(axis=1))
            
            self.train_scores.append(train_acc)
            self.val_scores.append(val_acc)
            
            gap = train_acc - val_acc
            
            # 更新状态监控器
            self.state_monitor. update(
                train_loss=1 - train_acc,
                val_loss=1 - val_acc,
                train_metric=train_acc,
                val_metric=val_acc
            )
            
            # 预热期后才开始自适应调整
            if total_trees > self.warmup_rounds:
                self._adapt_params_v2(total_trees, gap)
            
            # 记录参数历史
            self.param_history.append({
                'iteration': total_trees,
                'max_depth': self.current_max_depth,
                'min_child_weight': self.current_min_child_weight,
                'gamma': self.current_gamma,
                'reg_lambda': self.current_reg_lambda,
                'subsample': self.current_subsample,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'gap': gap
            })
            
            # 保存最佳模型
            if val_acc > best_val_score:
                best_val_score = val_acc
                best_iteration = total_trees
                rounds_without_improvement = 0
                # 深拷贝最佳模型
                self. best_model = self.model.copy()
            else:
                rounds_without_improvement += trees_per_round
            
            # 早停检查
            if rounds_without_improvement >= self.early_stopping_rounds:
                if self.verbose > 0:
                    print(f"\n  Early stopping at {total_trees} trees")
                break
            
            # 日志
            if self.verbose > 0:
                status = "WARMUP" if total_trees <= self.warmup_rounds else "ADAPT"
                print(
                    f"  [{status}] Trees: {total_trees:3d} | "
                    f"Train: {train_acc:.4f} | Val: {val_acc:.4f} | "
                    f"Gap: {gap:.4f} | "
                    f"Depth: {self.current_max_depth:2d}"
                )
        
        # 使用最佳模型
        if self.best_model is not None:
            self.model = self.best_model
        
        self.best_iteration = best_iteration
        self.best_val_score = best_val_score
        self.is_fitted = True
        
        if self.verbose > 0:
            print("-" * 60)
            print(f"Training completed!")
            print(f"  Best iteration: {best_iteration}")
            print(f"  Best val accuracy: {best_val_score:. 4f}")
        
        return self
    
    def _initialize_params_from_data(self):
        """根据数据特征初始化参数（更平衡的策略）"""
        noise_level = self.data_profile.get('estimated_noise_level', 0.1)
        n_samples = self.data_profile.get('n_samples', 1000)
        n_features = self.data_profile.get('n_features', 10)
        
        # 基础深度
        self.current_max_depth = self.initial_max_depth
        self.current_min_child_weight = 1.0
        
        # 只在极端情况下调整初始参数
        if noise_level > 0.25:
            self.current_max_depth = max(4, self.initial_max_depth - 1)
            self.current_min_child_weight = 2.0
        
        if n_samples < 300:
            self.current_max_depth = min(self.current_max_depth, 5)
            self.current_min_child_weight = max(2.0, self.current_min_child_weight)
            self.current_subsample = 0.9
        
        # 高维数据
        if n_features > 50:
            self.current_colsample = 0.8
        
        # 初始化正则化（保守值）
        self.current_gamma = 0.0
        self.current_reg_lambda = 1.0
        self.current_reg_alpha = 0.0
    
    def _get_xgb_params(self) -> Dict:
        """获取当前 XGBoost 参数"""
        params = {
            'objective': 'binary:logistic' if self.n_classes_ == 2 else 'multi:softprob',
            'eval_metric': 'logloss',
            'max_depth': int(self.current_max_depth),
            'min_child_weight': self.current_min_child_weight,
            'gamma': self. current_gamma,
            'reg_lambda': self.current_reg_lambda,
            'reg_alpha': self.current_reg_alpha,
            'subsample': self.current_subsample,
            'colsample_bytree': self.current_colsample,
            'learning_rate': self.learning_rate,
            'seed': self.random_state or 42,
            'verbosity': 0,
        }
        
        if self.n_classes_ > 2:
            params['num_class'] = self.n_classes_
        
        return params
    
    def _adapt_params_v2(self, iteration: int, current_gap: float):
        """自适应调整参数 V2 - 更平衡的策略"""
        
        # 获取历史趋势
        if len(self.val_scores) < 3:
            return
        
        recent_gaps = [self.train_scores[i] - self.val_scores[i] 
                       for i in range(-min(5, len(self.val_scores)), 0)]
        avg_gap = np.mean(recent_gaps)
        gap_trend = recent_gaps[-1] - recent_gaps[0] if len(recent_gaps) > 1 else 0
        
        recent_val = self.val_scores[-3:]
        val_improving = recent_val[-1] > recent_val[0]
        val_plateau = abs(recent_val[-1] - recent_val[0]) < 0.005
        
        noise_level = self.data_profile.get('estimated_noise_level', 0.1)
        n_samples = self.data_profile.get('n_samples', 1000)
        
        # 动态阈值
        adaptive_threshold = self.gap_threshold * (1 + noise_level * 0.5)
        
        # === 情况1: 明显过拟合（gap大且在增加）===
        if avg_gap > adaptive_threshold and gap_trend > 0.005:
            self._increase_regularization(strength=0.3)
        
        # === 情况2: 轻微过拟合（gap大但稳定）===
        elif avg_gap > adaptive_threshold and gap_trend <= 0.005:
            self._increase_regularization(strength=0.15)
        
        # === 情况3: 验证集在提升，gap可接受 ===
        elif val_improving and avg_gap <= adaptive_threshold:
            # 保持当前参数，不调整
            pass
        
        # === 情况4: 验证集平台期，可能欠拟合 ===
        elif val_plateau and avg_gap < adaptive_threshold * 0.5:
            # 如果验证集停滞且过拟合不严重，尝试增加复杂度
            if np.mean(recent_val) < 0.85: # 性能还有提升空间
                self._decrease_regularization(strength=0.2)
        
        # === 情况5: 验证集下降 ===
        elif not val_improving and len(self.val_scores) > 5:
            if self.val_scores[-1] < self.val_scores[-5]:
                # 验证集在下降，增加正则化
                self._increase_regularization(strength=0.25)
    
    def _increase_regularization(self, strength: float = 0.2):
        """增加正则化（减少过拟合）"""
        # 减小深度
        if self.current_max_depth > self.max_depth_range[0]: 
            depth_reduction = max(1, int(strength * 2))
            self.current_max_depth = max(
                self.max_depth_range[0],
                self.current_max_depth - depth_reduction
            )
        
        # 增加 min_child_weight
        self.current_min_child_weight = min(
            self.min_child_weight_range[1],
            self.current_min_child_weight * (1 + strength * 0.3)
        )
        
        # 增加 gamma
        self.current_gamma = min(1.0, self.current_gamma + strength * 0.1)
        
        # 增加 L2 正则化
        self. current_reg_lambda = min(5.0, self.current_reg_lambda * (1 + strength * 0.15))
        
        # 增加 dropout
        if strength > 0.25:
            self.current_subsample = max(0.7, self.current_subsample - 0.03)
    
    def _decrease_regularization(self, strength: float = 0.2):
        """减少正则化（增加模型复杂度）"""
        # 增加深度
        if self.current_max_depth < self.max_depth_range[1]:
            self.current_max_depth = min(
                self.max_depth_range[1],
                self.current_max_depth + 1
            )
        
        # 减少 min_child_weight
        self.current_min_child_weight = max(
            self.min_child_weight_range[0],
            self.current_min_child_weight * (1 - strength * 0.2)
        )
        
        # 减少 gamma
        self.current_gamma = max(0, self.current_gamma - strength * 0.05)
        
        # 减少 L2 正则化
        self. current_reg_lambda = max(0.5, self.current_reg_lambda * (1 - strength * 0.1))
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        X = np. asarray(X, dtype=np.float32)
        dtest = xgb.DMatrix(X)
        
        pred = self.model.predict(dtest)
        
        if self.n_classes_ == 2:
            return np.column_stack([1 - pred, pred])
        else:
            return pred
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测类别"""
        proba = self.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)
        return self.classes_[pred_indices]
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算准确率"""
        return accuracy_score(y, self. predict(X))
    
    def get_adaptation_history(self) -> Dict:
        """获取自适应历史"""
        return {
            'param_history': self.param_history,
            'train_scores': self.train_scores,
            'val_scores': self.val_scores,
            'data_profile': self.data_profile,
            'state_history': self.state_monitor.get_history(),
        }
    
    def plot_adaptation_history(self, figsize=(14, 10), save_path=None):
        """可视化自适应过程"""
        import matplotlib.pyplot as plt
        
        if not self.param_history:
            print("No history to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. 学习曲线
        ax = axes[0, 0]
        ax.plot([p['train_acc'] for p in self. param_history], label='Train', linewidth=2)
        ax.plot([p['val_acc'] for p in self.param_history], label='Val', linewidth=2)
        ax.axvline(x=self.warmup_rounds // self.adaptation_frequency, 
                   color='gray', linestyle='--', alpha=0.5, label='Warmup end')
        ax.set_xlabel('Adaptation Round')
        ax.set_ylabel('Accuracy')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Gap
        ax = axes[0, 1]
        gaps = [p['gap'] for p in self.param_history]
        ax.plot(gaps, color='red', linewidth=2)
        ax.axhline(y=self.gap_threshold, color='orange', linestyle='--', 
                   label=f'Threshold ({self.gap_threshold})')
        ax.set_xlabel('Adaptation Round')
        ax.set_ylabel('Gap')
        ax.set_title('Generalization Gap')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. max_depth
        ax = axes[0, 2]
        depths = [p['max_depth'] for p in self.param_history]
        ax.plot(depths, color='green', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Adaptation Round')
        ax.set_ylabel('max_depth')
        ax.set_title('Adaptive max_depth')
        ax.grid(True, alpha=0.3)
        
        # 4. min_child_weight
        ax = axes[1, 0]
        mcw = [p['min_child_weight'] for p in self.param_history]
        ax.plot(mcw, color='purple', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Adaptation Round')
        ax.set_ylabel('min_child_weight')
        ax.set_title('Adaptive min_child_weight')
        ax.grid(True, alpha=0.3)
        
        # 5. gamma
        ax = axes[1, 1]
        gammas = [p['gamma'] for p in self.param_history]
        ax.plot(gammas, color='blue', linewidth=2)
        ax.set_xlabel('Adaptation Round')
        ax.set_ylabel('gamma')
        ax.set_title('Adaptive gamma')
        ax.grid(True, alpha=0.3)
        
        # 6. reg_lambda
        ax = axes[1, 2]
        lambdas = [p['reg_lambda'] for p in self.param_history]
        ax.plot(lambdas, color='brown', linewidth=2)
        ax.set_xlabel('Adaptation Round')
        ax.set_ylabel('reg_lambda')
        ax.set_title('Adaptive reg_lambda')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig
    
    def get_params(self, deep=True):
        """获取参数（sklearn兼容）"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'initial_max_depth': self.initial_max_depth,
            'max_depth_range': self. max_depth_range,
            'min_child_weight_range': self.min_child_weight_range,
            'adaptation_frequency': self.adaptation_frequency,
            'gap_threshold': self.gap_threshold,
            'early_stopping_rounds': self. early_stopping_rounds,
            'validation_fraction': self.validation_fraction,
            'warmup_rounds': self.warmup_rounds,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }
    
    def __repr__(self):
        return f"AdaPruneXGB(n_estimators={self.n_estimators}, gap_threshold={self.gap_threshold})"