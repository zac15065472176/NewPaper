import pandas as pd
from pathlib import Path

# 读取你最新跑出来的带有早停机制的 raw_results 文件
csv_path = Path("results/tables/exp01_raw_results_20260417_023533.csv")
df = pd.read_csv(csv_path)

# 按 dataset 和 method 分组，计算 5 折的平均值
grouped = df.groupby(['dataset', 'method'])[['gap', 'accuracy', 'f1', 'train_time']].mean().unstack('method')

# 严格定义我们要在表格里展示的 7 个模型的列顺序 (6个基线 + 1个AdaPrune)
cols = ['RF_default', 'XGB_default', 'LGBM_default', 'XGB_tuned', 'XGB_early_stop', 'LGBM_early_stop', 'AdaPrune']

# 提取并保留 4 位小数
table6_f1 = grouped['f1'][cols].round(4)
table7_time = grouped['train_time'][cols].round(4)

# 增加一行 Average (均值)
table6_f1.loc['Average'] = table6_f1.mean().round(4)
table7_time.loc['Average'] = table7_time.mean().round(4)

# 导出为可以直接复制进 Word 的 CSV
table6_f1.to_csv("results/tables/Table6_F1_New.csv")
table7_time.to_csv("results/tables/Table7_Time_New.csv")

print("✅ 表 6 (F1) 和 表 7 (Time) 已经生成在 results/tables/ 目录下！请直接复制到 Word 中。")