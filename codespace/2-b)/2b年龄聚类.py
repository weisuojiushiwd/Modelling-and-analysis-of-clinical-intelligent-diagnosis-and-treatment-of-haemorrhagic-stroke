import pandas as pd
from sklearn.cluster import KMeans

# 读取Excel数据
data = pd.read_excel('竞赛发布数据\\表1-患者列表及临床信息.xlsx')

# 选择年龄作为聚类变量
X = data[['年龄']]

# 初始化K均值模型，将数据分成3个簇
kmeans = KMeans(n_clusters=3)

# 进行聚类
kmeans.fit(X)

# 将聚类结果添加到数据中
data['聚类结果'] = kmeans.labels_

# 将每个簇的数据保存到不同的Excel文件中
for cluster_label in range(3):
    cluster_data = data[data['聚类结果'] == cluster_label]
    cluster_data.to_excel(f'cluster_{cluster_label}.xlsx', index=False)
