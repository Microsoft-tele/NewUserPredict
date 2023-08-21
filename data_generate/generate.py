import pandas as pd
import torch
import colorama
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F


# 读取 CSV 文件，不将第一行视为列名
def generate_unknown(filepath: str, save_path: str):
    data = pd.read_csv(filepath)

    # 删除第0列和第2列
    data = data.drop(data.columns[[0, 2]], axis=1)

    # 去掉第一行（列名）
    data = data.iloc[1:]

    # 重置索引，以便重新设置正确的行索引
    data = data.reset_index(drop=True)

    # 将数据转换为NumPy数组
    data_numpy = data.values

    # ----------------------------------------------------------------
    # 创建MinMaxScaler对象
    # scaler = MinMaxScaler()

    # 训练缩放器并进行归一化
    # normalized_data = scaler.fit_transform(data_numpy.reshape(-1, 1))

    # 将归一化后的数据转回numpy（如果需要）
    # data_numpy = normalized_data.flatten().tolist()
    # -----------------------------------------------------------------

    # 如果使用Z-score方法：
    # 数据是一个(261423, 11)的NumPy数组

    # 转换为PyTorch张量
    data_tensor = torch.tensor(data_numpy, dtype=torch.float32)

    # 计算每列的均值和标准差
    mean = data_tensor.mean(dim=0)
    std = data_tensor.std(dim=0)

    # 应用Z-score归一化
    normalized_data = (data_tensor - mean) / std

    # 得到NumPy数组形式的归一化数据
    normalized_data_numpy = normalized_data.numpy()

    # 统一变量
    data_numpy = normalized_data_numpy

    print(normalized_data_numpy.shape)  # 形状与原始数据一致 (261423, 11)

    # --------------------------------------------------------------------------

    data_tensor = torch.from_numpy(data_numpy).float()
    torch.save(data_tensor, save_path)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print(colorama.Fore.RESET)


if __name__ == '__main__':
    generate_unknown("../dataset/train_unknown_cutTime.csv", "./train_unknown.pt")


