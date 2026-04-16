"""
AdaPrune XGBoost 优化版本 V3 (Native Callback Integration)

改进：通过原生 TrainingCallback 实现 O(1) 复杂度状态监控与底层参数热注入
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


class AdaPruneCallback(xgb.callback.TrainingCallback):
    """
    XGBoost 原生回调机制，用于在 C++ boosting 迭代流水线中进行极低开销的状态检测与剪枝参数干预。
    """
    def __init__(self, ap_model):
        self.ap = ap_model

    def after_iteration(self, model, epoch, evals_log):
        if not evals_log:
            return False

        # O(1) 复杂度指标提取，避免高开销的 predict()
        err_metric = 'error' if self.ap.n_classes_ == 2 else 'merror'
        train_err = evals_log['train'][err_metric][-1]
        val_err = evals_log['val'][err_metric][-1]

        train_acc = 1.0 - train_err
        val_acc = 1.0 - val_err
        gap = train_acc - val_acc

        # 更新历史测度
        self.ap.train_scores.append(train_acc)
        self.ap.val_scores.append(val_acc)

        self.ap.state_monitor.update(
            train_loss=train_err,
            val_loss=val_err,
            train_metric=train_acc,
            val_metric=val_acc
        )

        # 触发自适应惩罚控制
        if epoch > self.ap.warmup_rounds and epoch % self.ap.adaptation_frequency == 0:
            self.ap._adapt_params_v2(epoch, gap)

            # C++ 底层参数热更新
            model.set_param({
                'max_depth': int(self.ap.current_max_depth),
                'min_child_weight': float(self.ap.current_min_child_weight),
                'gamma': float(self.ap.current_gamma),
                'lambda': float(self.ap.current_reg_lambda),
                'subsample': float(self.ap.current_subsample)
            })

        # 记录拓扑参数演进轨迹
        self.ap.param_history.append({
            'iteration': epoch,
            'max_depth': self.ap.current_max_depth,
            'min_child_weight': self.ap.current_min_child_weight,
            'gamma': self.ap.current_gamma,
            'reg_lambda': self.ap.current_reg_lambda,
            'subsample': self.ap.current_subsample,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'gap': gap
        })

        return False # 维持流水线连续运转


class AdaPruneXGB:
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
        self.max_depth_range = max_depth_range
        self.min_child_weight_range = min_child_weight_range
        self.adaptation_frequency = adaptation_frequency
        self.gap_threshold = gap_threshold
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction
        self.warmup_rounds = warmup_rounds
        self.random_state = random_state
        self.verbose = verbose

        self.data_analyzer = DataProfileAnalyzer(random_state=random_state)
        self.state_monitor = StateMonitor(window_size=10)

        self.model: Optional[xgb.Booster] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_classes_: int = 0
        self.label_encoder: Optional[LabelEncoder] = None
        self.data_profile: Dict = {}
        self.is_fitted: bool = False

        self.current_max_depth: int = initial_max_depth
        self.current_min_child_weight: float = 1.0
        self.current_gamma: float = 0.0
        self.current_reg_lambda: float = 1.0
        self.current_reg_alpha: float = 0.0
        self.current_subsample: float = 1.0
        self.current_colsample: float = 1.0

        self.param_history: List[Dict] = []
        self.train_scores: List[float] = []
        self.val_scores: List[float] = []
        self.best_iteration: int = 0
        self.best_val_score: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None) -> 'AdaPruneXGB':

        if self.random_state is not None:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        if X_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_fraction,
                random_state=self.random_state, stratify=y
            )
        else:
            X_train, y_train = X, y
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = self.label_encoder.transform(y_val)

        if self.verbose > 0:
            print("=" * 60)
            print("AdaPrune XGB V3 (Native Callback) Training")
            print("=" * 60)

        self.data_profile = self.data_analyzer.analyze(X_train, y_train)
        self._initialize_params_from_data()

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        self.state_monitor.reset()
        self.param_history = []
        self.train_scores = []
        self.val_scores = []

        params = self._get_xgb_params()
        pruning_callback = AdaPruneCallback(self)

        # 激活单次连续训练与原生机制
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=self.early_stopping_rounds,
            callbacks=[pruning_callback],
            verbose_eval=False
        )

        self.is_fitted = True
        self.best_iteration = self.model.best_iteration
        self.best_val_score = self.val_scores[self.best_iteration] if self.val_scores else 0.0

        if self.verbose > 0:
            print("-" * 60)
            print("Training completed!")
            print(f"  Best iteration: {self.best_iteration}")
            print(f"  Best val accuracy: {self.best_val_score:.4f}")

        return self

    def _initialize_params_from_data(self):
        noise_level = self.data_profile.get('estimated_noise_level', 0.1)
        n_samples = self.data_profile.get('n_samples', 1000)
        n_features = self.data_profile.get('n_features', 10)

        self.current_max_depth = self.initial_max_depth
        self.current_min_child_weight = 1.0

        if noise_level > 0.25:
            self.current_max_depth = max(4, self.initial_max_depth - 1)
            self.current_min_child_weight = 2.0

        if n_samples < 300:
            self.current_max_depth = min(self.current_max_depth, 5)
            self.current_min_child_weight = max(2.0, self.current_min_child_weight)
            self.current_subsample = 0.9

        if n_features > 50:
            self.current_colsample = 0.8

        self.current_gamma = 0.0
        self.current_reg_lambda = 1.0
        self.current_reg_alpha = 0.0

    def _get_xgb_params(self) -> Dict:
        params = {
            'objective': 'binary:logistic' if self.n_classes_ == 2 else 'multi:softprob',
            # 引入 error/merror 以供 Callback 获取精度指标
            'eval_metric': ['logloss', 'error' if self.n_classes_ == 2 else 'merror'],
            'max_depth': int(self.current_max_depth),
            'min_child_weight': self.current_min_child_weight,
            'gamma': self.current_gamma,
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

        adaptive_threshold = self.gap_threshold * (1 + noise_level * 0.5)

        if avg_gap > adaptive_threshold and gap_trend > 0.005:
            self._increase_regularization(strength=0.3)
        elif avg_gap > adaptive_threshold and gap_trend <= 0.005:
            self._increase_regularization(strength=0.15)
        elif val_improving and avg_gap <= adaptive_threshold:
            pass
        elif val_plateau and avg_gap < adaptive_threshold * 0.5:
            if np.mean(recent_val) < 0.85:
                self._decrease_regularization(strength=0.2)
        elif not val_improving and len(self.val_scores) > 5:
            if self.val_scores[-1] < self.val_scores[-5]:
                self._increase_regularization(strength=0.25)

    def _increase_regularization(self, strength: float = 0.2):
        if self.current_max_depth > self.max_depth_range[0]:
            depth_reduction = max(1, int(strength * 2))
            self.current_max_depth = max(
                self.max_depth_range[0],
                self.current_max_depth - depth_reduction
            )
        self.current_min_child_weight = min(
            self.min_child_weight_range[1],
            self.current_min_child_weight * (1 + strength * 0.3)
        )
        self.current_gamma = min(1.0, self.current_gamma + strength * 0.1)
        self.current_reg_lambda = min(5.0, self.current_reg_lambda * (1 + strength * 0.15))
        if strength > 0.25:
            self.current_subsample = max(0.7, self.current_subsample - 0.03)

    def _decrease_regularization(self, strength: float = 0.2):
        if self.current_max_depth < self.max_depth_range[1]:
            self.current_max_depth = min(
                self.max_depth_range[1],
                self.current_max_depth + 1
            )
        self.current_min_child_weight = max(
            self.min_child_weight_range[0],
            self.current_min_child_weight * (1 - strength * 0.2)
        )
        self.current_gamma = max(0, self.current_gamma - strength * 0.05)
        self.current_reg_lambda = max(0.5, self.current_reg_lambda * (1 - strength * 0.1))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        X = np.asarray(X, dtype=np.float32)
        dtest = xgb.DMatrix(X)
        pred = self.model.predict(dtest, iteration_range=(0, self.best_iteration + 1))

        if self.n_classes_ == 2:
            return np.column_stack([1 - pred, pred])
        else:
            return pred

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)
        return self.classes_[pred_indices]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return accuracy_score(y, self.predict(X))

    def get_params(self, deep=True):
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'initial_max_depth': self.initial_max_depth,
            'max_depth_range': self.max_depth_range,
            'min_child_weight_range': self.min_child_weight_range,
            'adaptation_frequency': self.adaptation_frequency,
            'gap_threshold': self.gap_threshold,
            'early_stopping_rounds': self.early_stopping_rounds,
            'validation_fraction': self.validation_fraction,
            'warmup_rounds': self.warmup_rounds,
            'random_state': self.random_state,
            'verbose': self.verbose,
        }
    
    def __repr__(self):
        return f"AdaPruneXGB(n_estimators={self.n_estimators}, gap_threshold={self.gap_threshold})"