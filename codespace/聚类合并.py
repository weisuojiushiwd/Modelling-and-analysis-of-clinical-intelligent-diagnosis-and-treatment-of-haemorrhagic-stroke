import pandas as pd

# 读取三个残差数据文件
residual_df_0 = pd.read_excel('残差数据聚类0.xlsx')
residual_df_1 = pd.read_excel('残差数据聚类1.xlsx')
residual_df_2 = pd.read_excel('残差数据聚类2.xlsx')

# 合并三个数据框
combined_residual_df = pd.concat([residual_df_0, residual_df_1, residual_df_2], ignore_index=True)

# 按照ID列进行排序
combined_residual_df['ID'] = combined_residual_df['ID'].apply(lambda x: int(x[3:]))
combined_residual_df = combined_residual_df.sort_values(by='ID', ascending=True)

# 删除多余的ID列
combined_residual_df = combined_residual_df.drop(columns=['ID'])

# 将合并后的数据保存到一个新的Excel文件中
combined_residual_df.to_excel('合并的残差数据.xlsx', index=False)
