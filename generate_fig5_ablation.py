import matplotlib.pyplot as plt
import numpy as np

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 顺序：Static Prior, Trend Only, Gap Only, Hybrid, DataAware Only
configs = ['Static\nPrior', 'Trend\nOnly', 'Gap\nOnly', 'Hybrid', 'DataAware\nOnly']
accuracy = [0.8588, 0.8588, 0.8606, 0.8623, 0.8625]  #
gap = [0.0491, 0.0491, 0.0448, 0.0432, 0.0419]       #

# 颜色设置：基线用灰色，中间变体用蓝色，最终完全体用绿色
colors = ['#A5A5A5', '#A5A5A5', '#5B9BD5', '#70AD47', '#70AD47']

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Accuracy 对比
ax1 = axes[0]
x = np.arange(len(configs))
bars1 = ax1.bar(x, accuracy, color=colors, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
ax1.set_title('(a) Accuracy by Configuration', fontsize=12, fontweight='bold')
ax1.set_ylim(0.850, 0.870) # 针对新数据优化 Y 轴
ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=9)

# 添加数值标签
for bar, val in zip(bars1, accuracy):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# (b) Gap 对比
ax2 = axes[1]
bars2 = ax2.bar(x, gap, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Generalization Gap', fontsize=12, fontweight='bold')
ax2.set_title('(b) Gap by Configuration', fontsize=12, fontweight='bold')
ax2.set_ylim(0.040, 0.055) # 针对新数据优化 Y 轴
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=9)

# 添加数值标签
for bar, val in zip(bars2, gap):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('new_Figure5_Ablation.png', dpi=300)
plt.show()