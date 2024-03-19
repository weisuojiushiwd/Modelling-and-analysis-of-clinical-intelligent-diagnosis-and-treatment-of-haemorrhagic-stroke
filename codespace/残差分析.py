import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 读取Excel文件
df = pd.read_excel('合并的残差数据.xlsx')

# 提取ResidualError列
residual_error = df['ResidualError']

# 计算方差和平均值
variance = residual_error.var()
mean = residual_error.mean()

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(residual_error, bins=20, edgecolor='k', alpha=0.7, color='skyblue')
plt.title('Residual Error 分布', fontsize=15)
plt.xlabel('Residual Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 在图上添加方差和平均值文本
plt.text(50, 21, f'方差 = {variance:.2f}', fontsize=12, color='blue')
plt.text(50, 18, f'平均值 = {mean:.2f}', fontsize=12, color='blue')

plt.show()
