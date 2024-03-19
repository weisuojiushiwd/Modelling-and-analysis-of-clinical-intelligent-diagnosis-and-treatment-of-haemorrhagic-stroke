import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np
import joblib
from matplotlib.font_manager import FontProperties
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)
# 读取数据
df1 = pd.read_excel('竞赛发布数据\\表1-患者列表及临床信息.xlsx') 
df1 = df1.loc[:99, '年龄': '营养神经']

df2 = pd.read_excel('竞赛发布数据\\表2-患者影像信息血肿及水肿的体积及位置.xlsx')
df2 = df2.loc[:99, 'HM_volume': 'ED_Cerebellum_L_Ratio']

df3 = pd.read_excel('竞赛发布数据\\表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx')
df3 = df3.loc[:99, 'original_shape_Elongation': 'NCCT_original_firstorder_Variance']

# 合并数据
df = pd.concat([df1, df2, df3], axis=1)

# 读取目标变量
target_data = pd.read_excel('竞赛发布数据\\表1-患者列表及临床信息.xlsx')
df['90天mRS'] = target_data['90天mRS']

# 准备数据
y = df['90天mRS']
X = df.drop(columns=['90天mRS'])

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=66)

# 初始化XGBoost回归器
xgb_regressor = xgb.XGBRegressor(n_estimators=50,random_state=42)

# 训练模型
xgb_regressor.fit(X_train, y_train)
print("XGBoost Regressor training completed.")
xgb.plot_importance(xgb_regressor,height=1,max_num_features=20)
model_filename = 'xgboost3a_model.pkl'
joblib.dump(xgb_regressor, model_filename)
# 进行预测
y_pred_xgb = xgb_regressor.predict(X_test)

# 计算均方误差（MSE）
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print("Mean Squared Error (XGBoost):", mse_xgb)

# 计算平均绝对误差（MAE）
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
print("Mean Absolute Error (XGBoost):", mae_xgb)

# 计算均方根误差（RMSE）
rmse_xgb = np.sqrt(mse_xgb)
print("Root Mean Squared Error (XGBoost):", rmse_xgb)




# 可视化预测结果
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_xgb, color='blue')
plt.xlabel('实际值', fontproperties=font)
plt.ylabel('预测值', fontproperties=font)
plt.title('XGBoost 回归预测结果', fontproperties=font)
plt.show()
