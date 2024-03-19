import pandas as pd

# 读取聚类结果
cluster_0 = pd.read_excel('cluster_0.xlsx')
cluster_1 = pd.read_excel('cluster_1.xlsx')
cluster_2 = pd.read_excel('cluster_2.xlsx')

# 读取原始数据
original_data = pd.read_excel('竞赛发布数据\\数据.xlsx')

# 根据ID列将数据拆分成三个DataFrame
cluster_0_data = original_data[original_data['ID'].isin(cluster_0['ID'])]
cluster_1_data = original_data[original_data['ID'].isin(cluster_1['ID'])]
cluster_2_data = original_data[original_data['ID'].isin(cluster_2['ID'])]

# 保存到新的Excel文件
cluster_0_data.to_excel('cluster_0_data.xlsx', index=False)
cluster_1_data.to_excel('cluster_1_data.xlsx', index=False)
cluster_2_data.to_excel('cluster_2_data.xlsx', index=False)
