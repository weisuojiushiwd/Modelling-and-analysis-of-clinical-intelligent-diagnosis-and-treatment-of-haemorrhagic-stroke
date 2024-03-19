import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
font = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=12)

# 读取特征数据
df1 = pd.read_excel('竞赛发布数据\\表1-患者列表及临床信息.xlsx') 
df1 = df1.loc[:99, '年龄': '营养神经']

df2 = pd.read_excel('匹配结果.xlsx')
df2 = df2.loc[:99, 'HM_volume': 'ED_Cerebellum_L_Ratio']

df3 = pd.read_excel('竞赛发布数据\\表3-患者影像信息血肿及水肿的形状及灰度分布.xlsx')
df3 = df3.loc[:99, 'original_shape_Elongation': 'NCCT_original_firstorder_Variance']

# 合并数据
X = pd.concat([df1, df2, df3], axis=1)

# 读取目标变量
target_data = pd.read_excel('竞赛发布数据\\血肿扩张.xlsx')
y = target_data['血肿扩张']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=888)

# 初始化逻辑回归模型
logistic_regression = LogisticRegression(random_state=56666)

# 训练最终模型
logistic_regression.fit(X_train, y_train)
print("Logistic Regression Model training completed.")
# 画特征重要性柱状图
importances = logistic_regression.coef_[0]
features = X.columns

# 将特征重要性和特征名称对应起来
feature_importance = list(zip(features, importances))
feature_importance.sort(key=lambda x: abs(x[1]), reverse=False)
features, importances = zip(*feature_importance)

# 限制只显示前 20 个特征
top_features = features[:20]
top_importances = importances[:20]

plt.figure(figsize=(10, 6))
plt.barh(range(len(top_features)), top_importances, align='center')
plt.yticks(range(len(top_features)), top_features)
plt.xlabel('特征重要性')
plt.title('逻辑回归特征重要性 (Top 20)')
plt.show()

# 保存模型
model_filename = 'logistic_regression_model.pkl'
joblib.dump(logistic_regression, model_filename)

# 预测
y_pred = logistic_regression.predict(X_test)

# 计算准确度
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 计算Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# 计算Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# 计算F1分数
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1)

# 计算ROC曲线和AUC
probabilities = logistic_regression.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, probabilities)
auc = roc_auc_score(y_test, probabilities)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率', fontproperties=font)
plt.ylabel('真正率', fontproperties=font)
plt.title('逻辑回归ROC曲线', fontproperties=font)
plt.legend(loc="lower right", prop=font)
plt.show()
