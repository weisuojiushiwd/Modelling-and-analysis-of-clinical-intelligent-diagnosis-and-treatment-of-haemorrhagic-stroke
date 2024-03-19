import pandas as pd

# 读取第一个表
df1 = pd.read_excel('竞赛发布数据\\表2-患者影像信息血肿及水肿的体积及位置.xlsx')

# 提取第1行到第101行
selected_rows = df1.iloc[0:100]

# 读取第二个表
df2 = pd.read_excel('竞赛发布数据\\表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx')

# 根据流水号匹配相应的行
merged_data = pd.merge(selected_rows, df2, left_on='首次检查流水号', right_on='流水号', how='inner')

# 保存匹配到的行到新的表格
merged_data.to_excel('匹配结果.xlsx', index=False)
