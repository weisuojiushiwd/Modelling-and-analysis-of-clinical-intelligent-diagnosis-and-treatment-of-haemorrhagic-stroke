import pandas as pd
from scipy.stats import f_oneway

# 读取数据
data = pd.read_excel('竞赛发布数据\\2a.xlsx')
data = data.loc[:99]
independent = pd.read_excel('竞赛发布数据\\表1-患者列表及临床信息.xlsx') 
independent = independent.loc[:99, '脑室引流': '营养神经']

# 提取特征和目标变量
X = independent[['营养神经']]
y = data['ED_volume1'] - data['ED_volume0']  # 计算随访1与随访0之间的差值

# 执行方差分析
grouped_data = [y[X['营养神经'] == 0], y[X['营养神经'] == 1]]
f_statistic, p_value = f_oneway(*grouped_data)

# 输出方差分析结果
print(f'F-statistic: {f_statistic}')
print(f'p-value: {p_value}')

# 判断结果是否显著
alpha = 0.05
if p_value < alpha:
    print('在显著性水平0.05下，存在组间均值差异。')
else:
    print('在显著性水平0.05下，不存在组间均值差异。')
