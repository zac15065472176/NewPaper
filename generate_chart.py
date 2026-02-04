from graphviz import Digraph

# 创建有向图
dot = Digraph('AdaPrune', comment='AdaPrune Framework Overview')
dot.attr(rankdir='LR', size='10,8', dpi='300')  # 从左到右布局，高分辨率

# 设置通用节点样式 (矩形，填充颜色)
dot.attr('node', shape='rect', style='filled', fontname='Arial', fontsize='12', margin='0.2')

# 1. 定义节点 (按照原图颜色)

# 蓝色节点
dot.node('TrainData', 'Training\nData', fillcolor='#6baed6')

# 绿色节点
dot.node('ProfileAnalyzer', 'Data Profile\nAnalyzer', fillcolor='#74c476')
dot.node('GBDTTraining', 'GBDT\nTraining', fillcolor='#74c476')
dot.node('StateMonitor', 'State\nMonitor', fillcolor='#74c476')

# 紫色节点
dot.node('TrainedModel', 'Trained\nModel', fillcolor='#bc80bd')

# 红色节点 (Pruning Controller)
dot.node('PruningController', 'Pruning\nController', fillcolor='#fb6a4a', width='1.5', height='1.5')

# 浅红色节点 (策略列表) - 使用 HTML 标签模拟垂直排列的效果，或者单独定义节点
# 为了保持原图布局，我们把这四个策略定义为单独节点，并通过 invisible edge 强制垂直排列
with dot.subgraph(name='cluster_strategies') as c:
    c.attr(style='invis')  # 隐藏边框
    c.node('Hybrid', 'Hybrid', fillcolor='#fcae91')
    c.node('DataAware', 'Data-aware', fillcolor='#fcae91')
    c.node('TrendBased', 'Trend-based', fillcolor='#fcae91')
    c.node('GapBased', 'Gap-based', fillcolor='#fcae91')

    # 强制垂直布局
    c.edge('Hybrid', 'DataAware', style='invis')
    c.edge('DataAware', 'TrendBased', style='invis')
    c.edge('TrendBased', 'GapBased', style='invis')

# 2. 定义连接关系

# 左侧分支
dot.edge('TrainData', 'ProfileAnalyzer')
dot.edge('TrainData', 'GBDTTraining')

# 上方流程
dot.edge('ProfileAnalyzer', 'TrainedModel')

# 中间交互
# GBDT -> Profile Analyzer (带标签)
dot.edge('GBDTTraining', 'ProfileAnalyzer', label='• Noise Level\n• Sample Size\n• Class Overlap', fontsize='10')

# GBDT <-> State Monitor (双向箭头)
dot.edge('GBDTTraining', 'StateMonitor', dir='both', label='Adaptive\nFeedback', fontsize='10', style='dashed')

# State Monitor -> Pruning Controller (双向)
dot.edge('StateMonitor', 'PruningController', dir='both')

# State Monitor 下方的注释 (用一个不可见的节点或者 Label 属性实现，这里用 Label 挂在节点下稍微难看，我们用一个独立文本节点)
dot.node('MonitorNote', '• Gap\n• Trend\n• Overfit Score', shape='plain', style='', fillcolor='none', fontsize='10')
dot.edge('StateMonitor', 'MonitorNote', style='invis')  # 隐形线连在这个注释上

# 布局微调：让 Pruning Controller 和 策略组 稍微靠近一点
dot.edge('PruningController', 'Hybrid', style='invis')

# 保存并渲染
output_path = dot.render('adaprune_chart', view=False, format='png')
print(f"图片已生成: {output_path}")