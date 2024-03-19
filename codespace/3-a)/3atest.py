import pandas as pd
import joblib
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12
test_df1 = pd.read_excel('竞赛发布数据\\表1-患者列表及临床信息.xlsx') 
test_df1 = test_df1.loc[:159, '年龄': '营养神经']

test_df2 = pd.read_excel('竞赛发布数据\\表2-患者影像信息血肿及水肿的体积及位置.xlsx')
test_df2 = test_df2.loc[:159, 'HM_volume': 'ED_Cerebellum_L_Ratio']

test_df3 = pd.read_excel('竞赛发布数据\\数据.xlsx')
test_df3 = test_df3.loc[:159, 'original_shape_Elongation': 'NCCT_original_firstorder_Variance']

# 合并测试数据
test_df = pd.concat([test_df1, test_df2, test_df3], axis=1)

# 使用训练好的模型进行预测
model_filename = 'xgboost3a_model.pkl'
xgb_regressor = joblib.load(model_filename)

# 进行预测
y_pred_xgb = xgb_regressor.predict(test_df)

# 创建包含预测值的 DataFrame
predictions_df = pd.DataFrame({'预测值': y_pred_xgb})

# 将 DataFrame 保存为 Excel 文件
output_excel_filename = 'xgboost_predictions.xlsx'
predictions_df.to_excel(output_excel_filename, index=False)

# 打印保存成功的消息
print(f"预测结果已保存到 {output_excel_filename}")
# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.hist(y_pred_xgb, bins=30, color='blue', alpha=0.7)
plt.xlabel('预测值')
plt.ylabel('频数')
plt.title('预测值分布')
plt.show()
