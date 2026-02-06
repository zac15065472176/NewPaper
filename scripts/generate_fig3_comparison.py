import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 表4和表6的Average数据（来自仓库真实数据）
methods = ['XGB_def', 'XGB_tuned', 'RF', 'AP_hybrid', 'AP_data', 'AP_gap', 'AP_trend']
accuracy = [0.8967, 0.8937, 0.8972, 0.8754, 0.8748, 0.8702, 0.8648]  # 表4
gap = [0.1033, 0.1031, 0.1028, 0.0932, 0.0937, 0.0985, 0.1045]  # 表6

# 颜色：基线蓝色，AdaPrune橙色/绿色
colors_acc = ['#4472C4', '#4472C4', '#4472C4', '#C55A11', '#C55A11', '#C55A11', '#C55A11']
colors_gap = ['#4472C4', '#4472C4', '#4472C4', '#70AD47', '#70AD47', '#70AD47', '#70AD47']

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) 测试准确率对比
ax1 = axes[0]
x = np.arange(len(methods))
bars1 = ax1.bar(x, accuracy, color=colors_acc, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('(a) Test Accuracy Comparison', fontsize=12, fontweight='bold')
ax1.set_ylim(0.84, 0.92)
ax1.set_xticks(x)
ax1.set_xticklabels(methods, rotation=35, ha='right', fontsize=9)
ax1.axhline(y=0.89, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

# 添加数值标签
for bar, val in zip(bars1, accuracy):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=8)

# 图例
legend_elements1 = [Patch(facecolor='#4472C4', label='Baseline'),
                    Patch(facecolor='#C55A11', label='AdaPrune')]
ax1.legend(handles=legend_elements1, loc='lower left')
ax1.grid(axis='y', alpha=0.3)

# (b) 泛化差距对比
ax2 = axes[1]
bars2 = ax2.bar(x, gap, color=colors_gap, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Generalization Gap', fontsize=12)
ax2.set_title('(b) Generalization Gap Comparison', fontsize=12, fontweight='bold')
ax2.set_ylim(0, 0.14)
ax2.set_xticks(x)
ax2.set_xticklabels(methods, rotation=35, ha='right', fontsize=9)
ax2.axhline(y=0.10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='0.10 threshold')

# 添加数值标签
for bar, val in zip(bars2, gap):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=8)

# 图例
legend_elements2 = [Patch(facecolor='#4472C4', label='Baseline'),
                    Patch(facecolor='#70AD47', label='AdaPrune (lower is better)')]
ax2.legend(handles=legend_elements2, loc='upper right')
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/fig3_main_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("图3已保存到 results/figures/fig3_main_comparison.png")
print("\n数据核对：")
print("AP_hybrid Gap = 0.0932, XGB_def Gap = 0.1033")
print("改进幅度 = (0.1033 - 0.0932) / 0.1033 = 9.8%")