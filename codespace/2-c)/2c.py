import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, explained_variance_score, max_error

# 读取数据
data = pd.read_excel('竞赛发布数据\\2a.xlsx')
data = data.loc[:99]
independent = pd.read_excel('竞赛发布数据\\表1-患者列表及临床信息.xlsx') 
independent = independent.loc[:99, '脑室引流': '营养神经']

# 提取特征和目标变量
X = independent[['止血治疗']]
y = data['ED_volume4'] - data['ED_volume2']  # 计算随访1与随访0之间的差值

# 数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为PyTorch张量
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train.values.reshape(-1, 1))
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test.values.reshape(-1, 1))

# 构建带有批标准化的神经网络模型
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 64),
            nn.BatchNorm1d(64), 
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

# 初始化模型、损失函数和优化器
model = RegressionModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 50000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

# 在测试集上进行预测
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.squeeze().numpy()  # 将张量转为NumPy数组
    y_true = y_test.squeeze().numpy()

# 计算各种性能指标
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
explained_var = explained_variance_score(y_true, y_pred)
max_err = max_error(y_true, y_pred)

print(f'平均绝对误差 (MAE): {mae}')
print(f'R-squared (R2): {r2}')
print(f'解释方差得分: {explained_var}')
print(f'最大误差: {max_err}')
