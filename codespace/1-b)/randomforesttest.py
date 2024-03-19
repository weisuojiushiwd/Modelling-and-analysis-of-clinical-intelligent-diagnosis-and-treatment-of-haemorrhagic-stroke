import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
# 读取测试数据（假设测试数据保存在名为test_data.xlsx的文件中，数据读取部分不变）
test_df1 = pd.read_excel('竞赛发布数据\\表1-患者列表及临床信息.xlsx') 
test_df1 = test_df1.loc[:159, '年龄': '营养神经']

test_df2 = pd.read_excel('竞赛发布数据\\表2-患者影像信息血肿及水肿的体积及位置.xlsx')
test_df2 = test_df2.loc[:159, 'HM_volume': 'ED_Cerebellum_L_Ratio']

test_df3 = pd.read_excel('竞赛发布数据\\表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx')
test_df3 = test_df3.loc[:159, 'original_shape_Elongation': 'NCCT_original_firstorder_Variance']

# 合并测试数据
test_df = pd.concat([test_df1, test_df2, test_df3], axis=1)

# 使用训练好的模型进行预测概率
model_filename = 'random_forest_model.pkl'
rf_classifier = joblib.load(model_filename)

# 预测概率
probabilities = rf_classifier.predict_proba(test_df)
# 提取"会得血肿"的概率
positive_class_probability = probabilities[:, 1]

# 创建包含概率的 DataFrame
probability_df = pd.DataFrame({'会得血肿概率': positive_class_probability})

# 将 DataFrame 保存为 Excel 文件
output_excel_filename = 'blood_hemorrhage_probability.xlsx'
probability_df.to_excel(output_excel_filename, index=False)

# 打印保存成功的消息
print(f"概率已保存到 {output_excel_filename}")
# 打印概率结果
print("预测概率:", probabilities)
# 假设阈值为0.5
threshold = 0.5
binary_labels = (positive_class_probability > threshold).astype(int)
# 获取随机森林中每个决策树的参数
tree_params = [tree.get_params() for tree in rf_classifier.estimators_]

# 设置 StratifiedKFold 用于分层的5折交叉验证
cv = StratifiedKFold(n_splits=5)

# 初始化空列表用于存储每一折的AUC
auc_scores = []

# 初始化空列表用于存储每一折的FPR和TPR
mean_fpr = np.linspace(0, 1, 100)
tprs = []

# 获取每折的验证集和预测概率
for train_index, test_index in cv.split(test_df, binary_labels):
    X_train, X_test = test_df.iloc[train_index], test_df.iloc[test_index]
    y_train, y_test = binary_labels[train_index], binary_labels[test_index]

    rf_classifier.fit(X_train, y_train)
    probabilities = rf_classifier.predict_proba(X_test)[:, 1]

    # 计算AUC并保存
    auc = roc_auc_score(y_test, probabilities)
    auc_scores.append(auc)

    # 计算ROC曲线
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    tprs.append(np.interp(mean_fpr, fpr, tpr))

# 打印每折的AUC
for i, auc in enumerate(auc_scores):
    print(f'Fold {i+1} AUC: {auc:.2f}')

# 计算平均AUC
mean_auc = sum(auc_scores) / len(auc_scores)
print(f'平均AUC: {mean_auc:.2f}')

# 计算平均TPR
mean_tpr = np.mean(tprs, axis=0)

# 绘制每折的ROC曲线
plt.figure(figsize=(10, 6))
for i, tpr in enumerate(tprs):
    plt.plot(mean_fpr, tpr, lw=1, alpha=0.3, label=f'Fold {i+1} (AUC = {auc_scores[i]:.2f}')

plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean AUC = {mean_auc:.2f}', lw=2)


plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('5折交叉验证ROC曲线')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
