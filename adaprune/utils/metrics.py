"""
评估指标工具
"""

import numpy as np
from typing import Dict, List, Optional, Callable, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.model_selection import StratifiedKFold, KFold
import time


def compute_metrics(
    y_true:  np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    task: str = 'classification'
) -> Dict[str, float]:
    """
    计算评估指标
    
    Parameters
    ----------
    y_true : np.ndarray
        真实标签
    y_pred : np. ndarray
        预测标签
    y_proba :  np.ndarray, optional
        预测概率
    task : str
        任务类型
    
    Returns
    -------
    metrics : dict
        评估指标字典
    """
    metrics = {}
    
    if task == 'classification':
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        
        # 多分类处理
        n_classes = len(np.unique(y_true))
        average = 'binary' if n_classes == 2 else 'weighted'
        
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        if y_proba is not None:
            try:
                if n_classes == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[: , 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(
                        y_true, y_proba, multi_class='ovr', average='weighted'
                    )
                metrics['log_loss'] = log_loss(y_true, y_proba)
            except Exception: 
                pass
    
    else:  # regression
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np. sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['r2'] = r2_score(y_true, y_pred)
    
    return metrics


def cross_validate(
    model: Any,
    X:  np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    task: str = 'classification',
    random_state: int = 42,
    return_train_scores: bool = False,
    verbose: int = 0
) -> Dict[str, List[float]]: 
    """
    交叉验证
    
    Parameters
    ----------
    model : estimator
        模型对象（需要有fit和predict方法）
    X : np.ndarray
        特征矩阵
    y :  np.ndarray
        目标变量
    n_folds : int
        折数
    task : str
        任务类型
    random_state : int
        随机种子
    return_train_scores : bool
        是否返回训练集得分
    verbose : int
        日志详细程度
    
    Returns
    -------
    results : dict
        交叉验证结果
    """
    if task == 'classification': 
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else: 
        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    results = {
        'test_accuracy': [],
        'test_f1': [],
        'test_roc_auc':  [],
        'fit_time': [],
        'score_time': [],
    }
    
    if return_train_scores:
        results['train_accuracy'] = []
        results['train_f1'] = []
    
    for fold, (train_idx, test_idx) in enumerate(kfold. split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 训练
        start_time = time.time()
        
        # 克隆模型（简单实现）
        model_clone = model.__class__(**{
            k: v for k, v in model.__dict__.items()
            if not k.startswith('_') and not callable(v)
        })
        
        model_clone.fit(X_train, y_train)
        fit_time = time. time() - start_time
        
        # 评估
        start_time = time. time()
        y_pred = model_clone.predict(X_test)
        
        if hasattr(model_clone, 'predict_proba'):
            y_proba = model_clone.predict_proba(X_test)
        else:
            y_proba = None
        
        score_time = time.time() - start_time
        
        test_metrics = compute_metrics(y_test, y_pred, y_proba, task)
        
        results['test_accuracy']. append(test_metrics. get('accuracy', 0))
        results['test_f1'].append(test_metrics.get('f1', 0))
        results['test_roc_auc'].append(test_metrics.get('roc_auc', 0))
        results['fit_time'].append(fit_time)
        results['score_time'].append(score_time)
        
        if return_train_scores:
            y_train_pred = model_clone.predict(X_train)
            train_metrics = compute_metrics(y_train, y_train_pred, None, task)
            results['train_accuracy'].append(train_metrics.get('accuracy', 0))
            results['train_f1'].append(train_metrics.get('f1', 0))
        
        if verbose > 0:
            print(f"  Fold {fold + 1}/{n_folds}: "
                  f"Accuracy={test_metrics.get('accuracy', 0):.4f}, "
                  f"F1={test_metrics. get('f1', 0):.4f}")
    
    # 计算统计量
    summary = {}
    for key, values in results.items():
        summary[f'{key}_mean'] = np.mean(values)
        summary[f'{key}_std'] = np.std(values)
    
    results['summary'] = summary
    
    return results