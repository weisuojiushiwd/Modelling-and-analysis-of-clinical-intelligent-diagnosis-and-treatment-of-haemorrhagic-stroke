import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
# 读取合并后的治疗值数据
merged_data = pd.read_excel('merged_treatments.xlsx')

# 读取水肿体积数据
ed_volume_data = pd.read_excel('竞赛发布数据\\2a.xlsx')

# 合并数据，使用ID作为连接键
merged_data = pd.merge(merged_data, ed_volume_data[['ID', 'ED_volume0', 'ED_volume1']], on='ID', how='inner')
# 将缺失值替换为零
merged_data.fillna(0, inplace=True)
# 计算因变量
merged_data['Delta_ED_volume'] = merged_data['ED_volume1'] - merged_data['ED_volume0']
# 提取自变量和因变量
X = merged_data[['脑室引流', '止血治疗', '降颅压治疗', '降压治疗', '镇静、镇痛治疗', '止吐护胃', '营养神经']]  # 自变量，治疗值
y = merged_data['Delta_ED_volume']  # 因变量，水肿体积

# 创建XGBoost回归模型
model = xgb.XGBRegressor()
model.fit(X, y)

# 输出各指标的重要性
plot_importance(model)
plt.show()
