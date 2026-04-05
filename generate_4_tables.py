import pandas as pd
import glob
import os

# 1. 找到你最近运行生成的 exp01_raw_results 原始数据文件
files = glob.glob('results/tables/exp01_raw_results_*.csv')
if not files:
    print("找不到原始数据文件！请确认你在 results/tables/ 目录下有 exp01_raw_results 格式的文件。")
    exit()

latest_file = max(files, key=os.path.getctime)
print(f"正在读取最新的实验数据: {latest_file}\n")

df = pd.read_csv(latest_file)

# 2. 我们需要的算法顺序 (列顺序)
methods_order = ['RF_default', 'XGB_default', 'LGBM_default', 'XGB_tuned',
                 'AdaPrune_gap', 'AdaPrune_trend', 'AdaPrune_data', 'AdaPrune_hybrid']

# 3. 提取我们需要的 4 个核心指标
metrics = {
    'gap': 'Table 4: Generalization Gap (Lower is Better)',
    'accuracy': 'Table 5: Accuracy (Higher is Better)',
    'f1': 'Table 6: F1-Score (Higher is Better)',
    'train_time': 'Table 7: Training Time in Seconds (Lower is Better)'
}

# 4. 生成并打印四张大表
for metric_col, title in metrics.items():
    print("=" * 60)
    print(title)
    print("=" * 60)

    # 按数据集和方法计算 5 折的均值
    pivot_df = df.pivot_table(index='dataset', columns='method', values=metric_col, aggfunc='mean')

    # 按照我们规定的列排序
    # 注意：如果你的代码里方法名字有少许不同，请修改上面的 methods_order 列表匹配你的实际名字
    existing_cols = [m for m in methods_order if m in pivot_df.columns]
    pivot_df = pivot_df[existing_cols]

    # 增加 Average 行 (所有数据集的平均值)
    pivot_df.loc['Average'] = pivot_df.mean()

    # 格式化输出 (保留 4 位小数)
    print(pivot_df.round(4).to_string())
    print("\n")