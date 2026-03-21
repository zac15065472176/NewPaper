import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. 重新生成图3：整体性能对比 (Acc 和 Gap)
# ==========================================
methods = ['XGB_def', 'XGB_tuned', 'RF_def', 'AP_trend', 'AP_gap', 'AP_hybrid', 'AP_data']
# 填入 21 个数据集的最新平均真实数据
acc_scores = [0.8915, 0.8874, 0.8900, 0.8696, 0.8693, 0.8691, 0.8690]
gap_scores = [0.0906, 0.0739, 0.1095, 0.0604, 0.0598, 0.0577, 0.0558]

fig, (ax1, acc_ax2) = plt.subplots(1, 2, figsize=(14, 5))

# (a) Accuracy Plot
colors_acc = ['#4A7BC7']*3 + ['#D05A22']*4
bars1 = ax1.bar(methods, acc_scores, color=colors_acc, edgecolor='black')
ax1.set_ylim(0.84, 0.92)
ax1.set_ylabel('Accuracy')
ax1.set_title('(a) Test Accuracy Comparison')
ax1.axhline(y=0.8915, color='orange', linestyle='--', alpha=0.7) # XGB_def baseline
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)
ax1.tick_params(axis='x', rotation=30)

# (b) Gap Plot (Lower is better)
colors_gap = ['#4A7BC7']*3 + ['#60A647']*4
bars2 = acc_ax2.bar(methods, gap_scores, color=colors_gap, edgecolor='black')
acc_ax2.set_ylim(0.0, 0.13)
acc_ax2.set_ylabel('Generalization Gap')
acc_ax2.set_title('(b) Generalization Gap Comparison (Lower is Better)')
acc_ax2.axhline(y=0.0906, color='orange', linestyle='--', alpha=0.7) # XGB_def baseline
for bar in bars2:
    yval = bar.get_height()
    acc_ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.4f}', ha='center', va='bottom', fontsize=9)
acc_ax2.tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig('new_Figure3_Overall.png', dpi=300)
print("新图3 (new_Figure3_Overall.png) 已生成！")


# ==========================================
# 2. 重新生成图5：消融实验对比图
# ==========================================
ablation_methods = ['No_Adaptation\n(XGB_def)', 'Trend_Only', 'Gap_Only', 'Hybrid', 'Data_Aware']
ablation_acc = [0.8915, 0.8696, 0.8693, 0.8691, 0.8690]
ablation_gap = [0.0906, 0.0604, 0.0598, 0.0577, 0.0558]

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))

# (a) Ablation Acc
bars3 = ax3.bar(ablation_methods, ablation_acc, color=['#C25A24', '#5582C6', '#5582C6', '#5582C6', '#60A647'], edgecolor='black')
ax3.set_ylim(0.85, 0.91)
ax3.set_title('(a) Accuracy by Configuration')
for bar in bars3:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

# (b) Ablation Gap
bars4 = ax4.bar(ablation_methods, ablation_gap, color=['#C25A24', '#5582C6', '#5582C6', '#5582C6', '#60A647'], edgecolor='black')
ax4.set_ylim(0.04, 0.11)
ax4.set_title('(b) Gap by Configuration (Lower is Better)')
for bar in bars4:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, yval + 0.002, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

# 画那个绿色的下降箭头 (展示约38.4%的降低)
ax4.annotate('', xy=(4, 0.056), xytext=(0, 0.090),
            arrowprops=dict(arrowstyle="->", color='green', lw=2))
ax4.text(2, 0.075, '38.4% reduction', color='green', fontsize=12, fontweight='bold', ha='center')

plt.tight_layout()
plt.savefig('new_Figure5_Ablation.png', dpi=300)
print("新图5 (new_Figure5_Ablation.png) 已生成���")