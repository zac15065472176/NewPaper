import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# 设置
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# 颜色定义
colors = {
    'input': '#5B9BD5',      # 蓝色
    'profile': '#70AD47',    # 绿色
    'training': '#70AD47',   # 绿色
    'monitor': '#70AD47',    # 绿色
    'controller': '#9E7CC3', # 紫色
    'output': '#9E7CC3',     # 紫色
    'strategy': '#F4B183',   # 橙色
}

def draw_box(ax, x, y, w, h, text, color, fontsize=11):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', wrap=True)

def draw_arrow(ax, start, end, color='black'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# 绘制主要组件
# Training Data (输入)
draw_box(ax, 0.5, 3.5, 2, 1.2, 'Training\nData', colors['input'])

# Data Profile Analyzer
draw_box(ax, 3.5, 5.5, 2.5, 1.2, 'Data Profile\nAnalyzer', colors['profile'])

# GBDT Training
draw_box(ax, 3.5, 3.5, 2.5, 1.2, 'GBDT\nTraining', colors['training'])

# State Monitor
draw_box(ax, 7, 3.5, 2.2, 1.2, 'State\nMonitor', colors['monitor'])

# Pruning Controller
draw_box(ax, 10.2, 3.5, 2.3, 1.2, 'Pruning\nController', colors['controller'])

# Trained Model (输出)
draw_box(ax, 10.2, 5.5, 2.3, 1.2, 'Trained\nModel', colors['output'])

# 策略模块
strategies = ['Hybrid', 'Data-aware', 'Trend-based', 'Gap-based']
for i, strategy in enumerate(strategies):
    draw_box(ax, 10.2, 2.2 - i*0.7, 2.3, 0.55, strategy, colors['strategy'], fontsize=9)

# 绘制箭头连接
# Input -> Profile & Training
draw_arrow(ax, (2.5, 4.1), (3.5, 6.1))
draw_arrow(ax, (2.5, 4.1), (3.5, 4.1))

# Profile -> Training (虚线标注)
ax.annotate('', xy=(4.75, 5.5), xytext=(4.75, 4.7),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax.text(3.2, 5.1, '• Noise Level\n• Sample Size\n• Class Overlap', fontsize=8, va='top')

# Training -> Monitor
draw_arrow(ax, (6, 4.1), (7, 4.1))

# Monitor -> Controller
draw_arrow(ax, (9.2, 4.1), (10.2, 4.1))

# Controller -> Training (反馈环路)
ax.annotate('', xy=(4.75, 3.5), xytext=(4.75, 2.5),
            arrowprops=dict(arrowstyle='->', color='#C55A11', lw=2,
                           connectionstyle='arc3,rad=0'))
ax.annotate('', xy=(4.75, 2.5), xytext=(10.2, 2.5),
            arrowprops=dict(arrowstyle='-', color='#C55A11', lw=2))
ax.text(7.5, 2.7, 'Adaptive\nFeedback', fontsize=9, ha='center', color='#C55A11', style='italic')

# Training -> Output
draw_arrow(ax, (4.75, 4.7), (4.75, 5.5))
ax.annotate('', xy=(10.2, 6.1), xytext=(6, 6.1),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# Monitor标注
ax.text(7.5, 3.0, '• Gap\n• Trend\n• Overfit Score', fontsize=8, ha='left')

# 标题
ax.set_title('AdaPrune Framework Overview', fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('results/figures/fig1_framework.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()

print("图1已保存到 results/figures/fig1_framework.png")