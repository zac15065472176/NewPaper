import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 1. 终极超大画布，给巨型字体充足的排版空间
fig, ax = plt.subplots(1, 1, figsize=(28, 16))
ax.set_xlim(0, 28)
ax.set_ylim(0, 16)
ax.axis('off')

# 2. 国际顶级期刊经典淡雅学术配色 (保持高级感)
colors = {
    'input': ('#DAE8FC', '#6C8EBF'),
    'profile': ('#D5E8D4', '#82B366'),
    'training': ('#D5E8D4', '#82B366'),
    'monitor': ('#FFF2CC', '#D6B656'),
    'controller': ('#E1D5E7', '#9673A6'),
    'output': ('#F5F5F5', '#666666'),
    'strategy': ('#FFFFFF', '#B0B0B0'),
}

def draw_box(ax, x, y, w, h, text, color_tuple, fontsize=26):
    facecolor, edgecolor = color_tuple
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                         facecolor=facecolor, edgecolor=edgecolor, linewidth=3.0)
    ax.add_patch(box)
    # 强制拉开行距(linespacing)，彻底杜绝多行文字上下重叠
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='#222222', linespacing=1.6)

def draw_arrow(ax, start, end, color='#333333'):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=4.0, mutation_scale=40))

# 3. 超大方块尺寸，确保 26 号字体绝对不会溢出
box_w, box_h = 5.2, 2.6
# 彻底拉开列间距
col1 = 0.5
col2 = 7.5
col3 = 14.5
col4 = 21.5

# 彻底拉开行间距 (坐标整体上移，留出底部空间)
row_top = 12.5
row_mid = 8.0

# 绘制主流程模块
draw_box(ax, col1, row_mid, 4.0, box_h, 'Training\nData', colors['input']) # Input框稍微窄一点
draw_box(ax, col2, row_top, box_w, box_h, 'Data Prior\nAnalysis', colors['profile'])
draw_box(ax, col2, row_mid, box_w, box_h, 'GBDT\nTraining', colors['training'])
draw_box(ax, col3, row_mid, box_w, box_h, 'State\nMonitor', colors['monitor'])
draw_box(ax, col4, row_mid, box_w, box_h, 'Pruning\nController', colors['controller'])
draw_box(ax, col4, row_top, box_w, box_h, 'Trained\nModel', colors['output'])

# 绘制四种子策略模块
strategies = ['Hybrid', 'Data-aware', 'Trend-based', 'Gap-based']
strat_h = 1.0
strat_gap = 1.5
strat_start_y = 5.5  # 底部策略框起点
for i, strategy in enumerate(strategies):
    draw_box(ax, col4, strat_start_y - i*strat_gap, box_w, strat_h, strategy, colors['strategy'], fontsize=22)

# 4. 绘制所有箭头与文字 (精准错开)
# Input -> Profile
draw_arrow(ax, (col1 + 4.0, row_mid + box_h/2), (col2, row_top + box_h/2))
# Input -> Training
draw_arrow(ax, (col1 + 4.0, row_mid + box_h/2), (col2, row_mid + box_h/2))

# Profile -> Training
draw_arrow(ax, (col2 + box_w/2, row_top), (col2 + box_w/2, row_mid + box_h))
# 侧边说明文字，向右极大偏移，绝对不压线
ax.text(col2 + box_w/2 + 0.5, row_top - 0.7, '• Noise Level\n• Sample Size',
        fontsize=22, va='center', color='#333333', fontweight='bold', linespacing=1.5)

# Training -> Monitor
draw_arrow(ax, (col2 + box_w, row_mid + box_h/2), (col3, row_mid + box_h/2))

# Monitor -> Controller
draw_arrow(ax, (col3 + box_w, row_mid + box_h/2), (col4, row_mid + box_h/2))

# Monitor 提取信号标注 (悬挂在下方真空带，巨型字体)
ax.text(col3 + box_w/2, row_mid - 0.3, '• Gap    • Trend    • Overfit Score',
        fontsize=22, ha='center', va='top', color='#444444', fontweight='bold')

# Profile -> Trained Model (高空跨越线)
draw_arrow(ax, (col2 + box_w, row_top + box_h/2), (col4, row_top + box_h/2))

# Controller -> Training (底部核心反馈闭环线)
feedback_y = 7.1  # 完美卡在 row_mid(8.0) 和 strat_start(6.5) 中间
ax.plot([col4+0.8, col4+0.8, col2 + box_w/2, col2 + box_w/2],
        [row_mid, feedback_y, feedback_y, row_mid], color='#D86613', lw=4.0)
ax.annotate('', xy=(col2 + box_w/2, row_mid), xytext=(col2 + box_w/2, feedback_y+0.1),
            arrowprops=dict(arrowstyle='-|>', color='#D86613', lw=4.0, mutation_scale=40))
# 反馈线文字
ax.text(col3 + box_w/2, feedback_y - 0.2, 'Adaptive Feedback',
        fontsize=24, ha='center', va='top', color='#D86613', style='italic', fontweight='bold')

# 5. 主标题 (字号顶格)
ax.set_title('AdaPrune Framework Overview', fontsize=40, fontweight='bold', pad=30, color='#222222')

plt.tight_layout()
# 极高分辨率保存
plt.savefig('results/figures/fig1_framework_v4_Ultimate.png', dpi=400, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("✅ SCI级排版：终极无重叠、超大字号框架图 (fig1_framework_v4_Ultimate.png) 生成完毕！")