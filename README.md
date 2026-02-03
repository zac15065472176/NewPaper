# AdaPrune:  Online Adaptive Pruning for Gradient Boosting Decision Trees

## 概述

AdaPrune 是一个创新的自适应剪枝框架，用于梯度提升决策树（GBDT）。它在训练过程中动态调整剪枝参数，以实现更好的泛化性能。

## 核心特性

- 🔄 **在线自适应剪枝**：训练过程中动态调整 max_depth、min_samples_leaf 等参数
- 📊 **数据感知策略**：根据数据集特征（噪声、样本量、维度）调整适应速度
- 🎯 **多种策略**：Gap-Based、Trend-Based、Data-Aware、Hybrid
- ⚡ **高效实现**：计算开销仅增加约5-10%

## 安装

```bash
conda create -n adaprune python=3.10
conda activate adaprune
pip install -r requirements.txt
```

## 快速开始

```python
from adaprune import AdaPruneGBDT

# 创建模型
model = AdaPruneGBDT(
    n_estimators=100,
    strategy='hybrid',
    adaptation_frequency=5
)

# 训练
model.fit(X_train, y_train)

# 预测
y_pred = model. predict(X_test)

# 查看自适应历史
model.plot_adaptation_history()
```

## 项目结构

```
AdaPrune/
├── adaprune/          # 核心代码
├── experiments/       # 实验脚本
├── notebooks/         # Jupyter notebooks
├── data/              # 数据集
├── results/           # 实验结果
└── scripts/           # 工具脚本
```

## 引用

如果您使用了本项目，请引用：

```bibtex
@article{adaprune2024,
  title={AdaPrune: Online Adaptive Pruning for Gradient Boosting Decision Trees},
  author={Your Name},
  year={2024}
}
```