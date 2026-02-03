"""
可视化工具
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns


def plot_learning_curves(
    train_scores: List[float],
    val_scores: List[float],
    title: str = "Learning Curves",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制学习曲线
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    iterations = range(len(train_scores))
    
    ax.plot(iterations, train_scores, label='Train', linewidth=2, alpha=0.8)
    ax.plot(iterations, val_scores, label='Validation', linewidth=2, alpha=0.8)
    
    ax.fill_between(
        iterations,
        train_scores,
        val_scores,
        alpha=0.2,
        color='red',
        label='Gap'
    )
    
    ax. set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_comparison(
    results: pd.DataFrame,
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制方法对比图
    """
    fig, ax = plt. subplots(figsize=figsize)
    
    summary = results.groupby('method')[metric].agg(['mean', 'std']).reset_index()
    summary = summary. sort_values('mean', ascending=False)
    
    bars = ax.bar(
        summary['method'],
        summary['mean'],
        yerr=summary['std'],
        capsize=5,
        color=sns.color_palette("husl", len(summary)),
        edgecolor='black',
        linewidth=1.2
    )
    
    for bar, mean, std in zip(bars, summary['mean'], summary['std']):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height + std + 0.005,
            f'{mean:.3f}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax. set_xlabel('Method', fontsize=12)
    ax.set_ylabel(metric.capitalize(), fontsize=12)
    ax.set_title(f'{metric. capitalize()} Comparison', fontsize=14)
    ax.set_xticklabels(summary['method'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt. tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_feature_importance(
    importance:  np.ndarray,
    feature_names:  Optional[List[str]] = None,
    top_k: int = 20,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制特征重要性
    """
    n_features = len(importance)
    
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]
    
    # 排序取Top-K
    indices = np.argsort(importance)[::-1][:top_k]
    top_importance = importance[indices]
    top_names = [feature_names[i] for i in indices]
    
    fig, ax = plt. subplots(figsize=figsize)
    
    y_pos = np. arange(len(top_importance))
    
    bars = ax.barh(y_pos, top_importance, color=sns.color_palette("viridis", len(top_importance)))
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_k} Feature Importance', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt. tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_adaptation_comparison(
    histories: Dict[str, Dict],
    figsize: Tuple[int, int] = (14, 10),
    save_path:  Optional[str] = None
) -> plt.Figure:
    """
    对比不同策略的自适应过程
    """
    n_strategies = len(histories)
    fig, axes = plt. subplots(2, 2, figsize=figsize)
    
    colors = sns.color_palette("husl", n_strategies)
    
    # 1. 验证集得分对比
    ax = axes[0, 0]
    for (name, history), color in zip(histories. items(), colors):
        val_scores = history. get('val_scores', [])
        ax.plot(val_scores, label=name, color=color, linewidth=2, alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Score')
    ax.set_title('Validation Score Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Gap对比
    ax = axes[0, 1]
    for (name, history), color in zip(histories.items(), colors):
        train_scores = history. get('train_scores', [])
        val_scores = history.get('val_scores', [])
        if train_scores and val_scores:
            gaps = [t - v for t, v in zip(train_scores, val_scores)]
            ax.plot(gaps, label=name, color=color, linewidth=2, alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Gap (Train - Val)')
    ax.set_title('Generalization Gap Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. max_depth变化对比
    ax = axes[1, 0]
    for (name, history), color in zip(histories.items(), colors):
        param_history = history.get('param_history', [])
        if param_history:
            depths = [p. get('max_depth', 10) for p in param_history]
            ax.plot(depths, label=name, color=color, linewidth=2, alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('max_depth')
    ax.set_title('max_depth Adaptation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. min_samples_leaf变化对比
    ax = axes[1, 1]
    for (name, history), color in zip(histories.items(), colors):
        param_history = history.get('param_history', [])
        if param_history:
            min_samples = [p.get('min_samples_leaf', 1) for p in param_history]
            ax.plot(min_samples, label=name, color=color, linewidth=2, alpha=0.8)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('min_samples_leaf')
    ax.set_title('min_samples_leaf Adaptation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_scenario_heatmap(
    results: pd.DataFrame,
    metric:  str = 'accuracy',
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制场景对比热力图
    """
    pivot = results.pivot_table(
        values=metric,
        index='scenario',
        columns='method',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label':  metric.capitalize()}
    )
    
    ax.set_title(f'{metric.capitalize()} across Scenarios', fontsize=14)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Scenario', fontsize=12)
    
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_results_table(
    results:  pd.DataFrame,
    metrics: List[str] = ['accuracy', 'f1', 'roc_auc'],
    group_by: str = 'method',
    sort_by: Optional[str] = None,
    highlight_best: bool = True
) -> pd.DataFrame:
    """
    创建结果汇总表
    """
    agg_dict = {}
    for metric in metrics: 
        if metric in results.columns:
            agg_dict[metric] = ['mean', 'std']
    
    summary = results.groupby(group_by).agg(agg_dict).round(4)
    
    # 展平多级列名
    summary.columns = [f'{col[0]}_{col[1]}' for col in summary.columns]
    
    # 排序
    if sort_by and f'{sort_by}_mean' in summary.columns:
        summary = summary.sort_values(f'{sort_by}_mean', ascending=False)
    
    # 格式化为 mean±std
    formatted_summary = pd.DataFrame(index=summary.index)
    for metric in metrics:
        if f'{metric}_mean' in summary.columns:
            formatted_summary[metric] = summary. apply(
                lambda row: f"{row[f'{metric}_mean']:.4f}±{row[f'{metric}_std']:.4f}",
                axis=1
            )
    
    return formatted_summary


def plot_efficiency_comparison(
    results: pd.DataFrame,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    绘制效率对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # 1. 训练时间对比
    ax = axes[0]
    time_summary = results.groupby('method')['train_time']. agg(['mean', 'std']).reset_index()
    time_summary = time_summary.sort_values('mean')
    
    bars = ax.barh(
        time_summary['method'],
        time_summary['mean'],
        xerr=time_summary['std'],
        capsize=3,
        color=sns.color_palette("coolwarm", len(time_summary))
    )
    ax.set_xlabel('Training Time (s)')
    ax.set_title('Training Time Comparison')
    ax.grid(True, alpha=0.3, axis='x')
    
    # 2. 性能vs时间散点图
    ax = axes[1]
    method_summary = results.groupby('method').agg({
        'accuracy': 'mean',
        'train_time': 'mean'
    }).reset_index()
    
    colors = sns.color_palette("husl", len(method_summary))
    
    for i, row in method_summary. iterrows():
        ax.scatter(row['train_time'], row['accuracy'], 
                   s=150, c=[colors[i]], label=row['method'], edgecolors='black')
    
    ax.set_xlabel('Training Time (s)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Training Time')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path: 
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig