import matplotlib.pyplot as plt
import numpy as np

# 1. 严格对齐最新表 8 的消融实验数据
# 注意：X 轴标签我按照你原图加入了换行符，以防文字重叠
labels = ['No_Adaptation\n(XGB_def)', 'Trend_Only', 'Gap_Only', 'Hybrid', 'Data_Aware']

# 准确率数据 (Hybrid 已更新为最新 0.8754)
accuracy = [0.8915, 0.8696, 0.8693, 0.8754, 0.8690]

# 泛化间隙数据 (Hybrid 已更新为最新 0.0572)
gap = [0.0906, 0.0604, 0.0598, 0.0572, 0.0558]

# 2. 完全复刻你原图的柔和配色 (莫兰迪色系)
# 蓝色(基线), 橘色(单策略/混合策略), 绿色(数据感知)
colors = ['#A3C1E8', '#FDBE79', '#FDBE79', '#FDBE79', '#98DF8A']

# 3. 设置绘图参数 (1行2列的子图)
plt.rcParams['font.family'] = 'SimHei'  # 如果你需要中文支持，或者改回 'Times New Roman'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ==================== 左图：Accuracy ====================
x = np.arange(len(labels))
bars1 = ax1.bar(x, accuracy, color=colors, edgecolor='black', zorder=3)

ax1.set_title('(a) Accuracy by Configuration', fontsize=16)
ax1.set_ylabel('Test Accuracy', fontsize=14)
ax1.set_ylim(0.85, 0.91)  # 复刻你原图的 Y 轴范围
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=11)
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# 柱子上打上数值标签
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)


# ==================== 右图：Gap ====================
bars2 = ax2.bar(x, gap, color=colors, edgecolor='black', zorder=3)

ax2.set_title('(b) Gap by Configuration (Lower is Better)', fontsize=16)
ax2.set_ylabel('Generalization Gap', fontsize=14)
ax2.set_ylim(0.04, 0.11)  # 复刻你原图的 Y 轴范围
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=11)
ax2.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# 柱子上打上数值标签
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.001, f'{yval:.4f}', ha='center', va='bottom', fontsize=10)

# 4. 调整布局并保存
plt.tight_layout()
plt.savefig('Figure4_Ablation_Study_StyleFixed.png', dpi=300, bbox_inches='tight')
print("✅ 最新的消融实验图已生成！配色和双图结构已完美恢复为你的原始风格。")