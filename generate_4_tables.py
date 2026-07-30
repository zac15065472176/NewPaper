import pandas as pd
import glob
import os

# 1. 找到最新运行生成的 raw_results 原始数据文件
files = glob.glob('results/tables/exp01_raw_results_*.csv')
if not files:
    print("找不到原始数据文件！")
    exit()

latest_file = max(files, key=os.path.getctime)
print(f"正在读取最新的实验数据: {latest_file}\n")
df = pd.read_csv(latest_file)

# 2. 我们需要的算法顺序 (列顺序)
methods_order = ['RF_default', 'XGB_default', 'LGBM_default', 'XGB_tuned',
                 'XGB_early_stop', 'LGBM_early_stop', 'AdaPrune']


# 定义一个格式化函数：计算 均值 ± 标准差
def format_table(df, metric_col):
    df_mean = df.groupby(['dataset', 'method'])[metric_col].mean().unstack()
    df_std = df.groupby(['dataset', 'method'])[metric_col].std().unstack()

    df_combined = df_mean.copy()
    for col in df_mean.columns:
        df_combined[col] = df_mean[col].map('{:.4f}'.format) + " ± " + df_std[col].map('{:.4f}'.format)

    # 保留存在的列并排序
    existing_cols = [m for m in methods_order if m in df_combined.columns]
    df_combined = df_combined[existing_cols]

    # 增加 Average 行 (所有数据集的平均值 ± 平均标准差)
    avg_mean = df_mean.mean()
    avg_std = df_std.mean()

    avg_row = []
    for col in existing_cols:
        avg_row.append(f"{avg_mean[col]:.4f} ± {avg_std[col]:.4f}")

    df_combined.loc['Average'] = avg_row
    return df_combined


# 3. 提取我们需要的 4 个核心指标并生成带标准差的表
print("正在生成带标准差(±)的表格...")
table4_gap = format_table(df, 'gap')
table5_acc = format_table(df, 'accuracy')
table6_f1 = format_table(df, 'f1')
table7_time = format_table(df, 'train_time')

# 4. 导出为 CSV
table4_gap.to_csv("results/tables/Table4_Gap_with_std.csv")
table5_acc.to_csv("results/tables/Table5_Accuracy_with_std.csv")
table6_f1.to_csv("results/tables/Table6_F1_with_std.csv")
table7_time.to_csv("results/tables/Table7_Time_with_std.csv")

print("✅ 包含标准差的 4 张表格已经成功生成在 results/tables/ 目录下！")