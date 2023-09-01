import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# 加载鸢尾花数据集
data = load_iris()
X = data.data  # 特征矩阵
y = data.target  # 目标向量
print(y.shape)

# 标准化特征
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建KNN分类器，选择K值
k = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# 训练模型
knn_classifier.fit(X_train, y_train)

# 预测测试集
y_pred = knn_classifier.predict(X_test)

# 计算准确性
accuracy = accuracy_score(y_test, y_pred)
print(f'模型的准f1为: {accuracy * 100:.2f}%')
