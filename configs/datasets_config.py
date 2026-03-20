"""
AdaPrune 实验数据集配置
"""

# ==================== 核心数据集 (20+ 个真实数据集) ====================
CORE_DATASETS = {
    # ---------------- 原始保留的数据集 ----------------
    'adult': {'source': 'openml', 'openml_id': 1590, 'description': '人口收入预测', 'n_samples': 48842, 'n_features': 14, 'task': 'binary', 'difficulty': 'medium'},
    'bank-marketing': {'source': 'openml', 'openml_id': 1461, 'description': '银行营销预测', 'n_samples': 45211, 'n_features': 16, 'task': 'binary', 'difficulty': 'medium'},
    'credit-g': {'source': 'openml', 'openml_id': 31, 'description': '德国信用评估', 'n_samples': 1000, 'n_features': 20, 'task': 'binary', 'difficulty': 'medium'},
    'diabetes': {'source': 'openml', 'openml_id': 37, 'description': 'Pima糖尿病', 'n_samples': 768, 'n_features': 8, 'task': 'binary', 'difficulty': 'easy'},
    'ionosphere': {'source': 'openml', 'openml_id': 59, 'description': '电离层雷达', 'n_samples': 351, 'n_features': 34, 'task': 'binary', 'difficulty': 'medium'},
    'sonar': {'source': 'openml', 'openml_id': 40, 'description': '声纳信号分类', 'n_samples': 208, 'n_features': 60, 'task': 'binary', 'difficulty': 'hard'},
    'spambase': {'source': 'openml', 'openml_id': 44, 'description': '垃圾邮件分类', 'n_samples': 4601, 'n_features': 57, 'task': 'binary', 'difficulty': 'medium'},
    'vehicle': {'source': 'openml', 'openml_id': 54, 'description': '车辆轮廓分类', 'n_samples': 846, 'n_features': 18, 'task': 'multiclass', 'n_classes': 4, 'difficulty': 'medium'},
    'segment': {'source': 'openml', 'openml_id': 36, 'description': '图像分割', 'n_samples': 2310, 'n_features': 19, 'task': 'multiclass', 'n_classes': 7, 'difficulty': 'medium'},
    'satimage': {'source': 'openml', 'openml_id': 182, 'description': '卫星图像分类', 'n_samples': 6430, 'n_features': 36, 'task': 'multiclass', 'n_classes': 6, 'difficulty': 'medium'},
    'electricity': {'source': 'openml', 'openml_id': 151, 'description': '电价预测', 'n_samples': 45312, 'n_features': 8, 'task': 'binary', 'difficulty': 'medium'},
    'phoneme': {'source': 'openml', 'openml_id': 1489, 'description': '音素识别', 'n_samples': 5404, 'n_features': 5, 'task': 'binary', 'difficulty': 'easy'},
    'magic': {'source': 'openml', 'openml_id': 1120, 'description': 'MAGIC伽马望远镜', 'n_samples': 19020, 'n_features': 10, 'task': 'binary', 'difficulty': 'medium'},
    'eeg-eye-state': {'source': 'openml', 'openml_id': 1471, 'description': 'EEG眼睛状态', 'n_samples': 14980, 'n_features': 14, 'task': 'binary', 'difficulty': 'medium'},
    'waveform': {'source': 'openml', 'openml_id': 60, 'description': '波形分类', 'n_samples': 5000, 'n_features': 40, 'task': 'multiclass', 'n_classes': 3, 'difficulty': 'medium'},
    # ---------------- 成功下载的数据集 ----------------
    'amazon': {'source': 'openml', 'openml_id': 1457, 'description': '亚马逊员工访问', 'n_samples': 32769, 'n_features': 9, 'task': 'binary', 'difficulty': 'medium'},
    'dry-bean': {'source': 'openml', 'openml_id': 42585, 'description': '干豆分类', 'n_samples': 13611, 'n_features': 16, 'task': 'multiclass', 'n_classes': 7, 'difficulty': 'medium'},
    'letter': {'source': 'openml', 'openml_id': 6, 'description': '英文字母识别', 'n_samples': 20000, 'n_features': 16, 'task': 'multiclass', 'n_classes': 26, 'difficulty': 'hard'},
    'credit-default': {'source': 'openml', 'openml_id': 42477, 'description': '信用卡违约预测', 'n_samples': 30000, 'n_features': 23, 'task': 'binary', 'difficulty': 'medium'},
    # ---------------- 新增的3个100%可靠的数据集 ----------------
    'mushroom': {'source': 'openml', 'openml_id': 24, 'description': '蘑菇毒性预测', 'n_samples': 8124, 'n_features': 22, 'task': 'binary', 'difficulty': 'easy'},
    'tic-tac-toe': {'source': 'openml', 'openml_id': 50, 'description': '井字棋胜负', 'n_samples': 958, 'n_features': 9, 'task': 'binary', 'difficulty': 'easy'},
    'kc1': {'source': 'openml', 'openml_id': 1067, 'description': '软件缺陷预测', 'n_samples': 2109, 'n_features': 21, 'task': 'binary', 'difficulty': 'medium'},
}

# ==================== 合成数据集配置 (用于消融实验) ====================
SYNTHETIC_CONFIGS = {
    'clean_large': {'n_samples': 10000, 'n_features': 20, 'n_informative': 15, 'n_redundant': 3, 'noise': 0.0, 'flip_y': 0.0, 'description': '干净的大样本'},
    'clean_small': {'n_samples': 500, 'n_features': 20, 'n_informative': 15, 'n_redundant': 3, 'noise': 0.0, 'flip_y': 0.0, 'description': '干净的小样本'},
    'noisy_large': {'n_samples': 10000, 'n_features': 20, 'n_informative': 15, 'n_redundant': 3, 'noise': 0.3, 'flip_y': 0.1, 'description': '有噪声的大样本'},
    'noisy_small': {'n_samples': 500, 'n_features': 20, 'n_informative': 15, 'n_redundant': 3, 'noise': 0.3, 'flip_y': 0.15, 'description': '有噪声的小样本'},
    'high_dim': {'n_samples': 1000, 'n_features': 200, 'n_informative': 20, 'n_redundant': 30, 'noise': 0.1, 'flip_y': 0.05, 'description': '高维数据'},
    'imbalanced': {'n_samples': 5000, 'n_features': 20, 'n_informative': 15, 'n_redundant': 3, 'weights': [0.9, 0.1], 'noise': 0.1, 'flip_y': 0.05, 'description': '类别不平衡'},
}