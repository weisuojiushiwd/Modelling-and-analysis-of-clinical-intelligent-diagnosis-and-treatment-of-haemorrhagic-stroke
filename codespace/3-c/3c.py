import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
# 读取前448行数据
data_df = pd.read_excel('竞赛发布数据\\数据.xlsx', nrows=448)

# 选择你感兴趣的列
columns_of_interest = [
    '年龄', '性别', '高血压病史', '卒中病史', '糖尿病史', '房颤史', '冠心病史', '吸烟史', '饮酒史',
    '高压', '低压', '脑室引流', '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经',
    'HM_volume',  'ED_volume', '90天mRS'
]

# 计算相关性矩阵
correlation_matrix = data_df[columns_of_interest].corr(method='spearman')

# 使用热力图显示相关性矩阵
plt.figure(figsize=(16, 14))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title('相关性矩阵')
plt.show()
