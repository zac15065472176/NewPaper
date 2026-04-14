import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# 设置全局字体大小，适应论文排版
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14  # 全局基础字体调大

# 场景数据
scenarios = ['Clean\n(large)', 'Clean\n(small)', 'Noisy\n(large)', 'Noisy\n(small)', 'High-dim', 'Imbalanced']

# XGBoost默认参数的Gap
xgb_gap = [0.030, 0.100, 0.095, 0.182, 0.208, 0.060]
# AdaPrune_hybrid的Gap
adaprune_gap = [0.044, 0.090, 0.055, 0.110, 0.185, 0.065]

# 创建图形 (调大画布)
fig, ax = plt.subplots(figsize=(12, 6.5))

x = np.arange(len(scenarios))
width = 0.35

# 🎨 颜色改为高级论文常用的“淡色系” (Pastel Colors)
color_xgb = '#AEC7E8' # 淡蓝色
color_ap = '#FFBB78'  # 淡橙色

# 绘制柱状图，加上稍微明显一点的边框增加质感
bars1 = ax.bar(x - width / 2, xgb_gap, width, label='XGBoost', color=color_xgb, edgecolor='black', linewidth=1)
bars2 = ax.bar(x + width / 2, adaprune_gap, width, label='AdaPrune', color=color_ap, edgecolor='black', linewidth=1)

# 添加改进百分比标注 (字体调大)
for i, (xgb, ap) in enumerate(zip(xgb_gap, adaprune_gap)):
    if ap < xgb:  # AdaPrune更好
        improvement = (xgb - ap) / xgb * 100
        ax.annotate(f'-{improvement:.0f}%',
                    xy=(x[i] + width / 2, ap + 0.005),
                    ha='center', fontsize=13, color='#2CA02C', fontweight='bold') # 柔和的绿色
    elif ap > xgb:  # XGBoost更好
        ax.annotate(f'+{((ap - xgb) / xgb * 100):.0f}%',
                    xy=(x[i] + width / 2, ap + 0.005),
                    ha='center', fontsize=13, color='#D62728', fontweight='bold') # 柔和的红色

# 设置标签和标题 (字体大幅调大)
ax.set_ylabel('Generalization Gap', fontsize=16, fontweight='bold')
ax.set_title('Generalization Gap Across Different Scenarios', fontsize=18, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenarios, fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=14)
ax.set_ylim(0, 0.25)

# 添加背景虚线网格，辅助视觉
ax.grid(axis='y', alpha=0.4, linestyle='--')

plt.tight_layout()

# 确保保存路径存在（可选）
import os
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/fig4_scenario_analysis.png', dpi=300, bbox_inches='tight')

print("✅ 图4已保存到 results/figures/fig4_scenario_analysis.png")
plt.show()