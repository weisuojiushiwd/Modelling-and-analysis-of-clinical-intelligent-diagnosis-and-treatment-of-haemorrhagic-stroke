import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from scipy.optimize import curve_fit  
plt.rcParams['font.sans-serif'] = ['SimSun']  
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['font.size'] = 12  
# 读取数据  
df = pd.read_excel('cluster_0_data.xlsx')  
  
# 提取唯一的编号  
UniqueNums = df.iloc[1:, 0].unique()  
  
# 初始化列表  
RecheckTime = []  
EdemaData = []  
DiagnosisInfo = []  
  
for i in range(len(UniqueNums)):  
    a = np.where(df.iloc[:, 0] == UniqueNums[i])  
    DiagnosisInfo.append(df.iloc[a[0][0], 3:].values.astype(float))  
    RecheckTime.extend(df.iloc[a[0], 2].values.astype(float))  
    EdemaData.extend(df.iloc[a[0], 33].values.astype(float))  
  
  
# 使用高斯模型拟合水肿数据  
def gaussian_func(x, a, b, c):  
    return a * np.exp(-((x - b) / c)**2)  
  
params, covariance = curve_fit(gaussian_func, RecheckTime, EdemaData, p0=[1, 1000, 100], maxfev=5000)  
  
x_fit = np.linspace(1, 8000, 1000)  
y_fit = gaussian_func(x_fit, *params)  
# 计算残差  
EdemaFit = gaussian_func(np.array(RecheckTime), *params)  
ErrorA = []  
  
for i in range(len(UniqueNums)):  
    a = np.where(df.iloc[:, 0] == UniqueNums[i])[0] - 1  
    error = np.mean(np.abs(np.array(EdemaFit)[a] - np.array(EdemaData)[a])) * 0.001  
    ErrorA.append(error)  
  
# 创建包含残差的DataFrame  
residual_df = pd.DataFrame({'UniqueNums': UniqueNums, 'ResidualError': ErrorA})  
  
# 保存DataFrame到Excel文件  
residual_df.to_excel('残差数据聚类0.xlsx', index=False)  
  
# 绘制散点图  
plt.scatter(RecheckTime, EdemaData, marker='+')  
plt.plot(x_fit, y_fit, label='Gaussian Fit')  
plt.xlabel('时间')  
plt.ylabel('水肿/10^-3ml')  
plt.legend()  
plt.show()  
