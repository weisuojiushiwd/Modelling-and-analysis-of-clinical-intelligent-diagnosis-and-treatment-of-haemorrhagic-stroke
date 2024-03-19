import pandas as pd

# 假设您的数据存储在一个名为 data 的 DataFrame 中
# 如果数据不在 DataFrame 中，请将下面的 data 替换为您的实际数据来源
data = pd.read_excel('竞赛发布数据\\数据.xlsx')  # 读取数据

# 将缺失值填充为0
data.fillna(0, inplace=True)

# 提取 HM_volume 和 ED_volume 列
hm_volume = data['HM_volume']
ed_volume = data['ED_volume']

# 计算相关系数
correlation = hm_volume.corr(ed_volume, method='spearman')  # 使用斯皮尔曼秩相关系数

print(f'HM_volume 和 ED_volume 的相关系数为：{correlation:.4f}')
