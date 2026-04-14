import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小，适应论文排版
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14  # 全局基础字体调大

# 对应表 8 中的顺序
configs = ['No_Adaptation\n(XGB_def)', 'Trend_Only', 'Gap_Only', 'Hybrid', 'Data_Aware']

# 替换为你计算出的平均准确率数据 (来自你的 Table 8)
accuracy = [0.8915, 0.8696, 0.8693, 0.8691, 0.8690]
# 替换为你计算出的平均 Gap 数据 (来自你的 Table 8)
gap = [0.0906, 0.0604, 0.0598, 0.0577, 0.0558]

# 🎨 颜色设置：顶级学术期刊常用的“蓝-橙-绿”莫兰迪色系
color_baseline = '#AEC7E8'  # 淡蓝色 (代表传统的基线模型，客观、中性)
color_variants = '#FFBB78'  # 淡橙色 (代表过渡性的消融变体，温暖、探索)
color_best = '#98DF8A'      # 淡绿色 (代表最终的完全体 Data_Aware，积极、最佳)

# 颜色分配：第1个基线(蓝)，中间3个普通变体(橙)，最后1个最佳变体(绿)
colors_acc = [color_baseline, color_variants, color_variants, color_variants, color_best]
colors_gap = [color_baseline, color_variants, color_variants, color_variants, color_best]

# 创建图形 (调大画布)
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

# (a) Accuracy by Configuration
ax1 = axes[0]
x = np.arange(len(configs))
bars1 = ax1.bar(x, accuracy, color=colors_acc, edgecolor='black', linewidth=1)
ax1.set_ylabel('Test Accuracy', fontsize=16, fontweight='bold')
ax1.set_title('(a) Accuracy by Configuration', fontsize=18, fontweight='bold')
ax1.set_ylim(0.85, 0.91) # 调整 Y 轴范围以适应你的数据
ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=12)

# 添加数值标签 (字体调大)
for bar, val in zip(bars1, accuracy):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=12)

ax1.grid(axis='y', alpha=0.4, linestyle='--')

# (b) Gap by Configuration
ax2 = axes[1]
bars2 = ax2.bar(x, gap, color=colors_gap, edgecolor='black', linewidth=1)
ax2.set_ylabel('Generalization Gap', fontsize=16, fontweight='bold')
ax2.set_title('(b) Gap by Configuration (Lower is Better)', fontsize=18, fontweight='bold')
ax2.set_ylim(0.04, 0.11) # 调整 Y 轴范围以适应你的数据
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=12)

# 添加数值标签 (字体调大)
for bar, val in zip(bars2, gap):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
             f'{val:.4f}', ha='center', va='bottom', fontsize=12)

# 删除了绿色的标注线

ax2.grid(axis='y', alpha=0.4, linestyle='--')

plt.tight_layout()

# 确保保存路径存在（可选）
import os
os.makedirs('results/figures', exist_ok=True)
plt.savefig('results/figures/fig5_ablation.png', dpi=300, bbox_inches='tight')

print("✅ 图5已保存到 results/figures/fig5_ablation.png")
plt.show()