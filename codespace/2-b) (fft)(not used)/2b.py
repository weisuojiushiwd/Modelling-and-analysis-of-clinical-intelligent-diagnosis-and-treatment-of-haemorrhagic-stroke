import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)


# 读取数据
data = pd.read_excel('竞赛发布数据\\2a.xlsx')

# 提取ID列
ids = data['ID'].copy()  # 假设ID列的名称为'ID'
# 将缺失值替换为零
data.fillna(0, inplace=True)
# 删除ID列，以便在聚类时不影响
data.drop(columns=['ID'], inplace=True)
# 提取数据列的前缀
time_columns = [f'随访{i}距发病所差小时数' for i in range(13)]
volume_columns = [f'ED_volume{i}' for i in range(13)]

# 定义傅里叶变换函数
def fourier_transform(y):
    return np.fft.fft(y)

# 定义储存傅里叶变换结果的列表
fft_results = []


# 计算傅里叶变换结果
for index, row in data.iterrows():
    time_data = []
    volume_data = []

    for i in range(13):
        time_column = f'随访{i}距发病所差小时数'
        volume_column = f'ED_volume{i}'

        if time_column in data.columns and volume_column in data.columns:
            time_data.append(row[time_column])  # 将确切的时间数据添加到列表中
            volume_data.append(row[volume_column] * 1e-3)  # 将体积数据添加到列表中

    # 差分处理
    volume_data_diff = np.diff(volume_data)
    
    # 傅里叶变换
    fft_result = fourier_transform(volume_data_diff)
    fft_results.append(fft_result)
# 可视化拟合曲线和傅里叶变换结果
plt.figure(figsize=(12, 6))

# Plot all Fourier transform results
for result in fft_results:
    plt.plot(np.abs(result))

plt.xlabel('频率', fontproperties=font)
plt.ylabel('振幅', fontproperties=font)
plt.title('傅里叶变换结果', fontproperties=font)

plt.show()

# 将FFT结果转换为适合K-means聚类的形式
fft_results_reshaped = np.array([np.abs(result) for result in fft_results])

# 使用K-means聚类
n_clusters = 3  # 选择要聚类成的簇的数量
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
clusters = kmeans.fit_predict(fft_results_reshaped)

# 获取每个簇的中心点
cluster_centers = kmeans.cluster_centers_

# 绘制聚类结果和中心点
plt.figure(figsize=(12, 6))

# 设置颜色列表
colors = ['b', 'g', 'r']

for i in range(n_clusters):
    cluster_data = np.array(fft_results)[clusters == i]
    for result in cluster_data:
        plt.plot(np.abs(result), color=colors[i], label=f'Cluster {i}')

    # 画出中心点
    plt.plot(np.abs(cluster_centers[i]), 'o', color=colors[i])  # 'o' 表示圆点

plt.xlabel('频率', fontproperties=font)
plt.ylabel('振幅', fontproperties=font)
plt.title('均值聚类结果', fontproperties=font)

data['Cluster'] = clusters
data['ID'] = ids

# 输出每个聚类的数据行数
for cluster_label in np.unique(clusters):
    cluster_data = data[data['Cluster'] == cluster_label]
    print(f'Cluster {cluster_label}: {len(cluster_data)} rows')
for cluster_label in np.unique(clusters):
    cluster_data = data[data['Cluster'] == cluster_label]
    print(f'Cluster {cluster_label}: {len(cluster_data)} rows')
    print(f'IDs in Cluster {cluster_label}:')
    print(cluster_data['ID'])  # 假设ID列的名称为'ID'
# 保存结果到Excel
data.to_excel('聚类结果.xlsx', index=False)  # 保存到当前工作目录下的聚类结果.xlsx文件中
# 在新的图中连接每个簇的中心点
plt.figure(figsize=(12, 6))
for i in range(n_clusters):
    plt.plot(cluster_centers[i], 'o-', color=colors[i],label=f'Cluster {i}')  # 'o-' 表示圆点和线
plt.xlabel('频率', fontproperties=font)
plt.ylabel('振幅', fontproperties=font)
plt.title('簇中心点', fontproperties=font)
# 添加图例
plt.legend()
plt.show()


# 反傅里叶变换并打印结果
for i, cluster_center in enumerate(cluster_centers):
    inverse_transformed = np.fft.ifft(cluster_center).real  # 获取实部，这是一个实数数组

    # 创建一个时间数组，单位为小时
    time_interval = 1  # 因为每个时间点的单位是12小时
    x = np.arange(len(inverse_transformed)) * time_interval

    print(f'Cluster {i} 反傅里叶变换结果:')
    print(inverse_transformed)

    # 可选择将时域信号绘制出来
    plt.figure(figsize=(12, 6))
    plt.plot(x, inverse_transformed)  # 使用时间作为x轴
    plt.xlabel('时间（小时）', fontproperties=font)  # 更改x轴标签
    plt.ylabel('幅度', fontproperties=font)
    plt.title(f'Cluster {i} 反傅里叶变换结果', fontproperties=font)

    
    coeffs = np.polyfit(x, inverse_transformed, 3)
    fitted_curve = np.polyval(coeffs, x)
    plt.plot(x, fitted_curve, label='fiited', color='green')
    # 添加图例
    plt.legend()
    plt.show()

    
    print(f'Cluster {i} 拟合后的方程:')
    for j, coeff in enumerate(coeffs):
        print(f'x^{j}: {coeff}')

