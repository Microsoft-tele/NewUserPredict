import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from tools.config_file import NewUserPredictParams

params = NewUserPredictParams()


# # 加载鸢尾花数据集
# X = data.data  # 特征矩阵
# y = data.target  # 目标向量

def train_knn(df_train: pd.DataFrame):
    global knn_classifier
    X = df_train.iloc[:, 1:-1].values
    y = df_train.iloc[:, -1:].values
    y = y.ravel()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建KNN分类器，选择K值
    for k in range(1, 200):
        knn_classifier = KNeighborsClassifier(n_neighbors=k, n_jobs=6)

        # 训练模型
        knn_classifier.fit(X_train, y_train)

        # 预测测试集
        y_pred = knn_classifier.predict(X_test)

        # 计算准确性
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"当前k = {k}")
        print(f'\t--模型的准确率为: {accuracy * 100:.2f}%')
        print(f'\t--模型的f1为: {f1}')

    return knn_classifier


if __name__ == "__main__":
    df_train = pd.read_csv(params.train_knn_csv)
    df_test = pd.read_csv(params.test_knn_csv)
    X = df_test.iloc[:, 1:-1].values

    knn_classifier = train_knn(df_train)

    # y_predict = knn_classifier.predict(X)
    # uuid = df_test['uuid']
    # df = pd.DataFrame(y_predict, columns=['target'])
    # df['uuid'] = uuid
    # print(df)

    # df.to_csv(os.path.join(params.result_save_path, "knn_2023_9_2_0_16.csv"), index=False)