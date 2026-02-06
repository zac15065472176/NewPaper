import matplotlib.pyplot as plt
import numpy as np

# 设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# exp03_summary_20260118_080359.csv 的真实数据
configs = ['Full\nHybrid', 'Data-Aware\nOnly', 'Gap\nOnly', 'No\nAdaptation', 'Trend\nOnly']
accuracy = [0.8073, 0.8056, 0.8048, 0.8042, 0.8042]  # 仓库真实数据
gap = [0.0748, 0.0758, 0.0773, 0.0900, 0.0900]  # 仓库真实数据

# 颜色
colors_acc = ['#70AD47', '#5B9BD5', '#5B9BD5', '#C55A11', '#C55A11']
colors_gap = ['#70AD47', '#5B9BD5', '#5B9BD5', '#C55A11', '#C55A11']

# 创建图形
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (a) Accuracy by Configuration
ax1 = axes[0]
x = np.arange(len(configs))
bars1 = ax1.bar(x, accuracy, color=colors_acc, edgecolor='black', linewidth=0.5)
ax1.set_ylabel('Test Accuracy', fontsize=12)
ax1.set_title('(a) Accuracy by Configuration', fontsize=12, fontweight='bold')
ax1.set_ylim(0.790, 0.820)
ax1.set_xticks(x)
ax1.set_xticklabels(configs, fontsize=9)

# 添加数值标签
for bar, val in zip(bars1, accuracy):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

ax1.grid(axis='y', alpha=0.3)

# (b) Gap by Configuration
ax2 = axes[1]
bars2 = ax2.bar(x, gap, color=colors_gap, edgecolor='black', linewidth=0.5)
ax2.set_ylabel('Generalization Gap', fontsize=12)
ax2.set_title('(b) Gap by Configuration', fontsize=12, fontweight='bold')
ax2.set_ylim(0.05, 0.10)
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=9)

# 添加数值标签
for bar, val in zip(bars2, gap):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# 添加17%改进标注
ax2.annotate('', xy=(0, 0.0748), xytext=(3, 0.0900),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax2.text(1.5, 0.082, '17% reduction', fontsize=11, color='green',
         fontweight='bold', style='italic')

ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/fig5_ablation.png', dpi=300, bbox_inches='tight')
plt.show()

print("图5已保存到 results/figures/fig5_ablation.png")
print("\n数据核对（来自exp03_summary_20260118_080359.csv）：")
print("Full_hybrid: Acc=0.8073, Gap=0.0748")
print("No_adaptation: Acc=0.8042, Gap=0.0900")
print("改进幅度 = (0.0900 - 0.0748) / 0.0900 = 16.9% ≈ 17%")