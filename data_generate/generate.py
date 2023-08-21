import pandas as pd
import torch
import colorama
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

from config import Config

config = Config()


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

    print(data_numpy)

    # print(colorama.Fore.LIGHTGREEN_EX)
    # print("Convert dataset successfully!!!")
    # print(colorama.Fore.RESET)


if __name__ == '__main__':
    generate_unknown(config.train_unknown_csv, config.train_unknown_pt)
