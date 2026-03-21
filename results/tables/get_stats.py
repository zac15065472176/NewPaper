import pandas as pd
df = pd.read_csv('exp01_raw_results_20260321_020001.csv')
metrics = ['accuracy', 'gap', 'f1']
methods = df['method'].unique()
for metric in metrics:
    print(f"\n--- 统计指标: {metric} ---")
    mean_df = df.groupby(['dataset', 'method'])[metric].mean().unstack()
    best_counts = {m: 0 for m in methods}
    worst_counts = {m: 0 for m in methods}
    for dataset in mean_df.index:
        row = mean_df.loc[dataset]
        if metric == 'gap':
            best_method = row.idxmin()
            worst_method = row.idxmax()
        else:
            best_method = row.idxmax()
            worst_method = row.idxmin()
        best_counts[best_method] += 1
        worst_counts[worst_method] += 1
    print("Best 次数:", best_counts)
    print("Worst 次数:", worst_counts)