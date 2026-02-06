import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 模拟数据
np.random.seed(42)
iterations = np.arange(0, 200, 5)
n_points = len(iterations)

# (a) 学习曲线
train_acc = 0.70 + 0.25 * (1 - np.exp(-iterations/50)) + np.random.normal(0, 0.01, n_points)
val_acc = 0.70 + 0.18 * (1 - np.exp(-iterations/60)) + np.random.normal(0, 0.015, n_points)
train_acc = np.clip(train_acc, 0.65, 0.95)
val_acc = np.clip(val_acc, 0.65, 0.90)

# (b) Gap
gap = train_acc - val_acc

# (c) max_depth 自适应调整
max_depth = np.ones(n_points) * 6
max_depth[iterations >= 40] = 5
max_depth[iterations >= 140] = 4

# (d) min_child_weight 自适应调整
min_child_weight = np.ones(n_points) * 1.0
min_child_weight[iterations >= 50] = 1.0
min_child_weight[iterations >= 80] = 1.5
min_child_weight[iterations >= 110] = 2.0
min_child_weight[iterations >= 150] = 2.5

# 创建图形
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# (a) Learning Curves
ax1 = axes[0, 0]
ax1.plot(iterations, train_acc, 'b-', linewidth=2, label='Train')
ax1.plot(iterations, val_acc, 'r-', linewidth=2, label='Validation')
ax1.fill_between(iterations, val_acc, train_acc, alpha=0.3, color='red')
ax1.axvline(x=30, color='gray', linestyle='--', alpha=0.7, label='Warmup end')
ax1.set_xlabel('Boosting Iteration', fontsize=11)
ax1.set_ylabel('Accuracy', fontsize=11)
ax1.set_title('(a) Learning Curves', fontsize=12, fontweight='bold')
ax1.legend(loc='lower right')
ax1.set_ylim(0.65, 1.0)
ax1.grid(True, alpha=0.3)

# (b) Generalization Gap
ax2 = axes[0, 1]
ax2.plot(iterations, gap, 'r-', linewidth=2)
ax2.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, label='Threshold (0.05)')
ax2.axvline(x=30, color='gray', linestyle='--', alpha=0.7, label='Warmup end')
ax2.set_xlabel('Boosting Iteration', fontsize=11)
ax2.set_ylabel('Gap (Train - Val)', fontsize=11)
ax2.set_title('(b) Generalization Gap', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.set_ylim(0, 0.15)
ax2.grid(True, alpha=0.3)

# (c) Adaptive max_depth
ax3 = axes[1, 0]
ax3.step(iterations, max_depth, 'g-', linewidth=2, where='post', marker='o', markersize=4)
ax3.axvline(x=30, color='gray', linestyle='--', alpha=0.7, label='Warmup end')
ax3.annotate('Detect overfitting\n→ reduce depth', xy=(45, 5.5), xytext=(60, 6.3),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))
ax3.set_xlabel('Boosting Iteration', fontsize=11)
ax3.set_ylabel('max_depth', fontsize=11)
ax3.set_title('(c) Adaptive max_depth', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right')
ax3.set_ylim(3, 7)
ax3.grid(True, alpha=0.3)

# (d) Adaptive min_child_weight
ax4 = axes[1, 1]
ax4.step(iterations, min_child_weight, 'm-', linewidth=2, where='post', marker='o', markersize=4)
ax4.axvline(x=30, color='gray', linestyle='--', alpha=0.7, label='Warmup end')
ax4.set_xlabel('Boosting Iteration', fontsize=11)
ax4.set_ylabel('min_child_weight', fontsize=11)
ax4.set_title('(d) Adaptive min_child_weight', fontsize=12, fontweight='bold')
ax4.legend(loc='upper left')
ax4.set_ylim(0, 3.5)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/fig2_adaptation_process.png', dpi=300, bbox_inches='tight')
plt.show()

print("图2已保存到 results/figures/fig2_adaptation_process.png")