import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import time

# 导入你自己的模块 (请根据你的实际路径调整)
from adaprune.core.adaprune_gbdt import AdaPruneGBDT
# 假设你有一个加载数据的方法，或者你自己用 pandas 读入
from adaprune.utils.data_loader import load_dataset


def run_beta_sensitivity():
    # 我们只跑论文里指定的两个典型数据集
    datasets = ['synthetic_noisy_small', 'ionosphere']

    # 论文里要测的 beta 值列表
    beta_values = [0.1, 0.3, 0.5, 0.7, 0.9]

    # 存储最终结果的字典
    results = []

    print("🚀 开始运行平滑系数 Beta 的灵敏性分析...")

    for ds_name in datasets:
        print(f"\n========================================")
        print(f"📦 正在处理数据集: {ds_name}")

        # 加载数据 (替换为你自己的加载方式)
        X, y = load_dataset(ds_name)

        for beta in beta_values:
            print(f"  👉 正在测试 Beta = {beta} ...", end=" ")

            # 按照论文设定：5折分层交叉验证
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            fold_acc = []
            fold_gap = []

            for train_idx, test_idx in skf.split(X, y):
                X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train_full, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # 初始化你的模型
                # 注意：这里 strategy 必须用 'trend' (Trend-based)
                # 并且传入当前的 beta 值（请确保你的类支持这个参数！）
                model = AdaPruneGBDT(
                    strategy='trend',
                    gap_threshold=0.05,  # 固定阈值
                    ema_beta=beta,  # <--- 核心：传入当前循环的平滑系数
                    random_state=42
                )

                # 训练模型 (内部会自动切分20%做验证集)
                model.fit(X_train_full, y_train_full)

                # 预测
                y_pred_train = model.predict(X_train_full)
                y_pred_test = model.predict(X_test)

                # 计算指标
                acc_train = accuracy_score(y_train_full, y_pred_train)
                acc_test = accuracy_score(y_test, y_pred_test)
                gap = acc_train - acc_test

                fold_acc.append(acc_test)
                fold_gap.append(gap)

            # 计算当前 beta 下的 5折平均值
            mean_acc = np.mean(fold_acc)
            mean_gap = np.mean(fold_gap)

            print(f"完成! Accuracy: {mean_acc:.4f}, Gap: {mean_gap:.4f}")

            # 保存到结果列表
            results.append({
                'Dataset': ds_name,
                'Beta': beta,
                'Accuracy': mean_acc,
                'Gap': mean_gap
            })

    # 将结果转换为易读的表格格式并打印，方便你直接抄到论文的 Table 7 里
    df_results = pd.DataFrame(results)
    print("\n✅ 实验跑完啦！下面是你需要的论文表格数据：\n")

    # 透视表格式化输出
    pivot_acc = df_results.pivot(index='Dataset', columns='Beta', values='Accuracy')
    pivot_gap = df_results.pivot(index='Dataset', columns='Beta', values='Gap')

    print("--- Accuracy 准确率 ---")
    print(pivot_acc.round(4))
    print("\n--- Generalization Gap 泛化差距 ---")
    print(pivot_gap.round(4))


if __name__ == "__main__":
    run_beta_sensitivity()