"""
AdaPrune GBDT 主模型

自适应剪枝梯度提升决策树
"""

import numpy as np
from typing import Optional, Literal, Dict, List, Tuple, Union
from sklearn. tree import DecisionTreeRegressor
from sklearn. model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings

from . data_analyzer import DataProfileAnalyzer
from .state_monitor import StateMonitor
from .pruning_controller import PruningController, PruningParams

warnings.filterwarnings('ignore')


class AdaPruneGBDT: 
    """
    AdaPrune:  自适应剪枝梯度提升决策树
    
    核心创新：
    1. 在训练过程中动态调整剪枝参数
    2. 根据数据特征指导自适应策略
    3. 多种自适应策略可选
    
    Parameters
    ----------
    n_estimators : int, default=100
        树的数量
    learning_rate :  float, default=0.1
        学习率
    strategy : str, default='hybrid'
        自适应策略 ['gap_based', 'trend_based', 'data_aware', 'hybrid']
    adaptation_frequency : int, default=5
        参数调整频率
    task : str, default='classification'
        任务类型 ['classification', 'regression']
    validation_fraction :  float, default=0.2
        验证集比例
    early_stopping_rounds : int, optional
        早停轮数
    random_state : int, optional
        随机种子
    verbose : int, default=0
        日志详细程度
    
    Attributes
    ----------
    trees : list
        训练好的决策树列表
    classes_ : ndarray
        类别标签（分类任务）
    n_classes_ : int
        类别数量
    data_profile : dict
        数据画像
    is_fitted : bool
        是否已训练
    
    Examples
    --------
    >>> from adaprune import AdaPruneGBDT
    >>> model = AdaPruneGBDT(n_estimators=100, strategy='hybrid')
    >>> model.fit(X_train, y_train)
    >>> y_pred = model. predict(X_test)
    >>> model.plot_adaptation_history()
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        strategy:  str = 'hybrid',
        adaptation_frequency: int = 5,
        task:  Literal['classification', 'regression'] = 'classification',
        validation_fraction: float = 0.2,
        early_stopping_rounds: Optional[int] = None,
        initial_max_depth: int = 10,
        initial_min_samples_leaf: int = 1,
        random_state: Optional[int] = None,
        verbose:  int = 0
    ):
        # 基础参数
        self.n_estimators = n_estimators
        self. learning_rate = learning_rate
        self.strategy = strategy
        self.adaptation_frequency = adaptation_frequency
        self.task = task
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds
        self.initial_max_depth = initial_max_depth
        self.initial_min_samples_leaf = initial_min_samples_leaf
        self.random_state = random_state
        self.verbose = verbose
        
        # 组件初始化
        self.data_analyzer = DataProfileAnalyzer(random_state=random_state)
        self.state_monitor = StateMonitor()
        self.pruning_controller = PruningController(
            strategy=strategy,
            adaptation_frequency=adaptation_frequency,
            total_iterations=n_estimators
        )
        
        # 模型状态
        self. trees:  List[List[DecisionTreeRegressor]] = []
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.init_prediction:  Union[float, np.ndarray] = 0.0
        self.data_profile: Dict = {}
        self. is_fitted: bool = False
        self.label_encoder: Optional[LabelEncoder] = None
        
        # 训练历史
        self. train_scores_: List[float] = []
        self.val_scores_: List[float] = []
        self.best_iteration_: int = 0
    
    def fit(
        self,
        X: np. ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> 'AdaPruneGBDT':
        """
        训练模型
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            训练特征
        y :  np.ndarray, shape (n_samples,)
            训练标签
        X_val : np.ndarray, optional
            验证特征
        y_val : np.ndarray, optional
            验证标签
        
        Returns
        -------
        self :  AdaPruneGBDT
            训练好的模型
        """
        # 设置随机种子
        if self.random_state is not None: 
            np.random. seed(self.random_state)
        
        # 数据类型转换
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        
        # 处理标签编码
        if self.task == 'classification': 
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder. classes_
            self. n_classes_ = len(self.classes_)
        
        # 划分验证集
        if X_val is None:
            stratify = y if self.task == 'classification' else None
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=self.validation_fraction,
                random_state=self. random_state,
                stratify=stratify
            )
        else:
            X_train, y_train = X, y
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val)
            if self.task == 'classification':
                y_val = self.label_encoder. transform(y_val)
        
        # === Step 1: 数据分析 ===
        if self.verbose > 0:
            print("=" * 60)
            print("AdaPrune Training")
            print("=" * 60)
            print("\nStep 1: Analyzing dataset...")
        
        self.data_profile = self.data_analyzer.analyze(X_train, y_train)
        
        # 初始化剪枝参数
        initial_params = PruningParams(
            max_depth=self.initial_max_depth,
            min_samples_leaf=self.initial_min_samples_leaf
        )
        self.pruning_controller = PruningController(
            strategy=self.strategy,
            initial_params=initial_params,
            adaptation_frequency=self. adaptation_frequency,
            total_iterations=self.n_estimators
        )
        self.pruning_controller.initialize_from_data(self.data_profile)
        
        if self.verbose > 0:
            print(f"  Samples: {self.data_profile['n_samples']}")
            print(f"  Features: {self.data_profile['n_features']}")
            print(f"  Classes: {self.n_classes_ if self.task == 'classification' else 'N/A'}")
            print(f"  Estimated noise: {self.data_profile['estimated_noise_level']:.3f}")
            print(f"  Initial max_depth: {self. pruning_controller.current_params.max_depth}")
            print(f"  Strategy: {self.strategy}")
        
        # === Step 2: 迭代训练 ===
        if self.verbose > 0:
            print("\nStep 2: Training with adaptive pruning...")
            print("-" * 60)
        
        # 初始化预测值
        F_train, F_val = self._initialize_predictions(y_train, len(y_val))
        
        # 重置监控器
        self. state_monitor.reset()
        self.trees = []
        self.train_scores_ = []
        self.val_scores_ = []
        
        best_val_score = -np.inf
        best_iteration = 0
        rounds_without_improvement = 0
        
        for iteration in range(self.n_estimators):
            # 获取当前剪枝参数
            current_params = self.pruning_controller.adapt(
                state=self.state_monitor.get_state(),
                data_profile=self.data_profile,
                iteration=iteration
            )
            
            # 计算负梯度（伪残差）
            residuals = self._compute_residuals(y_train, F_train)
            
            # 训练基学习器
            trees_this_round = self._fit_trees(
                X_train, residuals, current_params
            )
            self.trees.append(trees_this_round)
            
            # 更新预测值
            F_train = self._update_predictions(F_train, X_train, trees_this_round)
            F_val = self._update_predictions(F_val, X_val, trees_this_round)
            
            # 计算性能指标
            train_loss, train_metric = self._evaluate(F_train, y_train)
            val_loss, val_metric = self._evaluate(F_val, y_val)
            
            self.train_scores_. append(train_metric)
            self.val_scores_.append(val_metric)
            
            # 计算模型复杂度
            complexity = sum(t.get_n_leaves() for t in trees_this_round)
            
            # 更新状态监控器
            self.state_monitor.update(
                train_loss=train_loss,
                val_loss=val_loss,
                train_metric=train_metric,
                val_metric=val_metric,
                model_complexity=complexity
            )
            
            # 早停检查
            if val_metric > best_val_score:
                best_val_score = val_metric
                best_iteration = iteration
                rounds_without_improvement = 0
            else: 
                rounds_without_improvement += 1
            
            if self.early_stopping_rounds: 
                if rounds_without_improvement >= self.early_stopping_rounds:
                    if self.verbose > 0:
                        print(f"\n  Early stopping at iteration {iteration}")
                    break
            
            # 日志输出
            if self.verbose > 0 and (iteration % 10 == 0 or iteration == self.n_estimators - 1):
                state = self.state_monitor.get_state()
                print(
                    f"  Iter {iteration:3d} | "
                    f"Train: {train_metric:.4f} | "
                    f"Val: {val_metric:.4f} | "
                    f"Gap: {state['gap']:.4f} | "
                    f"Depth: {current_params. max_depth: 2d} | "
                    f"MinLeaf: {current_params.min_samples_leaf:3d}"
                )
        
        self.best_iteration_ = best_iteration
        self.is_fitted = True
        
        if self.verbose > 0:
            print("-" * 60)
            print(f"Training completed!")
            print(f"  Best iteration: {best_iteration}")
            print(f"  Best validation score: {best_val_score:.4f}")
            print(f"  Total trees: {len(self.trees)}")
        
        return self
    
    def _initialize_predictions(
        self,
        y_train: np.ndarray,
        n_val:  int
    ) -> Tuple[np. ndarray, np. ndarray]:
        """初始化预测值"""
        n_train = len(y_train)
        
        if self.task == 'classification':
            # 使用log-odds初始化
            class_counts = np.bincount(y_train. astype(int), minlength=self.n_classes_)
            self.init_prediction = np.log(
                (class_counts + 1) / (len(y_train) + self.n_classes_)
            )
            F_train = np. tile(self.init_prediction, (n_train, 1))
            F_val = np.tile(self. init_prediction, (n_val, 1))
        else:
            self.init_prediction = np.mean(y_train)
            F_train = np.full(n_train, self.init_prediction)
            F_val = np.full(n_val, self.init_prediction)
        
        return F_train, F_val
    
    def _compute_residuals(
        self,
        y:  np.ndarray,
        F: np.ndarray
    ) -> np.ndarray:
        """计算负梯度（伪残差）"""
        if self.task == 'classification':
            probs = self._softmax(F)
            residuals = np.zeros_like(F)
            for k in range(self. n_classes_):
                residuals[: , k] = (y == k).astype(float) - probs[:, k]
            return residuals
        else:
            return y - F
    
    def _fit_trees(
        self,
        X:  np.ndarray,
        residuals:  np.ndarray,
        params: PruningParams
    ) -> List[DecisionTreeRegressor]: 
        """训练一轮的决策树"""
        trees = []
        sklearn_params = params.to_sklearn_params()
        
        if self.task == 'classification': 
            for k in range(self. n_classes_):
                tree = DecisionTreeRegressor(
                    **sklearn_params,
                    random_state=self. random_state
                )
                tree.fit(X, residuals[:, k])
                trees. append(tree)
        else:
            tree = DecisionTreeRegressor(
                **sklearn_params,
                random_state=self.random_state
            )
            tree.fit(X, residuals)
            trees.append(tree)
        
        return trees
    
    def _update_predictions(
        self,
        F:  np.ndarray,
        X: np.ndarray,
        trees: List[DecisionTreeRegressor]
    ) -> np.ndarray:
        """更新预测值"""
        if self.task == 'classification':
            for k, tree in enumerate(trees):
                F[: , k] += self.learning_rate * tree.predict(X)
        else: 
            F += self. learning_rate * trees[0].predict(X)
        return F
    
    def _softmax(self, F: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_F = np.exp(F - np.max(F, axis=1, keepdims=True))
        return exp_F / np. sum(exp_F, axis=1, keepdims=True)
    
    def _evaluate(
        self,
        F: np.ndarray,
        y: np.ndarray
    ) -> Tuple[float, float]:
        """评估模型"""
        if self. task == 'classification':
            probs = self._softmax(F)
            # 交叉熵损失
            eps = 1e-10
            loss = -np.mean([
                np.log(probs[i, int(y[i])] + eps) 
                for i in range(len(y))
            ])
            # 准确率
            preds = np.argmax(probs, axis=1)
            metric = accuracy_score(y, preds)
        else: 
            loss = mean_squared_error(y, F)
            metric = -loss  # 负MSE，越大越好
        
        return float(loss), float(metric)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（仅分类任务）
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            输入特征
        
        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
            类别概率
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted.  Call fit() first.")
        
        if self.task != 'classification': 
            raise ValueError("predict_proba is only for classification tasks")
        
        X = np.asarray(X, dtype=np.float32)
        n_samples = X. shape[0]
        
        F = np.tile(self.init_prediction, (n_samples, 1))
        
        for trees_round in self.trees:
            for k, tree in enumerate(trees_round):
                F[:, k] += self.learning_rate * tree.predict(X)
        
        return self._softmax(F)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            输入特征
        
        Returns
        -------
        predictions : np.ndarray
            预测结果
        """
        if not self. is_fitted: 
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np. float32)
        
        if self.task == 'classification':
            probs = self. predict_proba(X)
            pred_indices = np.argmax(probs, axis=1)
            return self.classes_[pred_indices]
        else:
            F = np.full(len(X), self.init_prediction)
            for trees_round in self.trees:
                F += self.learning_rate * trees_round[0].predict(X)
            return F
    
    def score(self, X:  np.ndarray, y: np.ndarray) -> float:
        """
        计算得分
        
        Parameters
        ----------
        X : np. ndarray
            输入特征
        y : np.ndarray
            真实标签
        
        Returns
        -------
        score : float
            准确率（分类）或负MSE（回归）
        """
        if self.task == 'classification': 
            return accuracy_score(y, self.predict(X))
        else:
            return -mean_squared_error(y, self. predict(X))
    
    def get_adaptation_history(self) -> Dict: 
        """
        获取自适应历史记录
        
        Returns
        -------
        history : dict
            包含参数历史、状态历史和数据画像
        """
        return {
            'param_history': self. pruning_controller.get_history(),
            'state_history':  self.state_monitor.get_history(),
            'data_profile': self.data_profile,
            'train_scores': self. train_scores_,
            'val_scores': self.val_scores_,
        }
    
    def plot_adaptation_history(
        self,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[str] = None
    ):
        """
        可视化自适应过程
        
        Parameters
        ----------
        figsize : tuple
            图像大小
        save_path : str, optional
            保存路径
        """
        import matplotlib.pyplot as plt
        
        history = self.get_adaptation_history()
        param_history = history['param_history']
        state_history = history['state_history']
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        iterations = [p['iteration'] for p in param_history]
        
        # 1. 学习曲线
        ax = axes[0, 0]
        ax.plot(state_history['train_metrics'], label='Train', alpha=0.8, linewidth=2)
        ax.plot(state_history['val_metrics'], label='Validation', alpha=0.8, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Metric')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 泛化差距
        ax = axes[0, 1]
        gaps = [t - v for t, v in zip(
            state_history['train_metrics'],
            state_history['val_metrics']
        )]
        ax.plot(gaps, color='red', alpha=0.8, linewidth=2)
        ax.axhline(y=0.05, color='orange', linestyle='--', label='Threshold')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gap (Train - Val)')
        ax.set_title('Generalization Gap')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. max_depth变化
        ax = axes[0, 2]
        depths = [p['max_depth'] for p in param_history]
        ax.plot(iterations, depths, marker='o', markersize=3, 
                color='green', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('max_depth')
        ax.set_title('Adaptive max_depth')
        ax.grid(True, alpha=0.3)
        
        # 4. min_samples_leaf变化
        ax = axes[1, 0]
        min_samples = [p['min_samples_leaf'] for p in param_history]
        ax. plot(iterations, min_samples, marker='o', markersize=3,
                color='purple', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('min_samples_leaf')
        ax.set_title('Adaptive min_samples_leaf')
        ax.grid(True, alpha=0.3)
        
        # 5. 正则化参数变化
        ax = axes[1, 1]
        gammas = [p['gamma'] for p in param_history]
        lambdas = [p['reg_lambda'] for p in param_history]
        ax.plot(iterations, gammas, label='gamma', alpha=0.8, linewidth=2)
        ax.plot(iterations, lambdas, label='reg_lambda', alpha=0.8, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value')
        ax.set_title('Regularization Parameters')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 6. 模型复杂度
        ax = axes[1, 2]
        ax.plot(state_history['complexities'], color='brown', 
                alpha=0.8, linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total Leaves')
        ax.set_title('Model Complexity')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt. savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
        
        return fig
    
    def get_feature_importance(self, importance_type: str = 'gain') -> np.ndarray:
        """
        获取特征重要性
        
        Parameters
        ----------
        importance_type :  str
            重要性类型 ('gain' 或 'split')
        
        Returns
        -------
        importance : np.ndarray
            特征重要性
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        n_features = self. trees[0][0].n_features_in_
        importance = np.zeros(n_features)
        
        for trees_round in self. trees:
            for tree in trees_round: 
                if importance_type == 'gain':
                    importance += tree.feature_importances_
                else: 
                    # 统计分裂次数
                    tree_features = tree.tree_.feature
                    for f in tree_features: 
                        if f >= 0:
                            importance[f] += 1
        
        # 归一化
        importance = importance / importance.sum()
        
        return importance
    
    def summary(self) -> str:
        """
        模型摘要
        
        Returns
        -------
        summary : str
            模型信息摘要
        """
        lines = [
            "=" * 60,
            "AdaPrune GBDT Summary",
            "=" * 60,
            f"Task: {self.task}",
            f"Strategy: {self.strategy}",
            f"n_estimators: {self.n_estimators}",
            f"learning_rate: {self.learning_rate}",
            f"adaptation_frequency: {self. adaptation_frequency}",
            "-" * 60,
        ]
        
        if self.is_fitted:
            lines.extend([
                f"Fitted: Yes",
                f"Total trees trained: {len(self.trees)}",
                f"Best iteration: {self. best_iteration_}",
                f"Final train score: {self. train_scores_[-1]:.4f}",
                f"Final val score:  {self.val_scores_[-1]:.4f}",
                f"Final max_depth: {self. pruning_controller. current_params.max_depth}",
                f"Final min_samples_leaf:  {self.pruning_controller.current_params.min_samples_leaf}",
            ])
        else:
            lines.append("Fitted: No")
        
        lines.append("=" * 60)
        
        return "\n". join(lines)
    
    def __repr__(self) -> str:
        return (
            f"AdaPruneGBDT(n_estimators={self.n_estimators}, "
            f"learning_rate={self.learning_rate}, "
            f"strategy='{self.strategy}')"
        )