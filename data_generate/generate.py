import math

import pandas as pd
import torch
import colorama
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

from tools.config_file import NewUserPredictParams
from dataset import processing_time_stamp, normalize

params = NewUserPredictParams()


def generate_unknown():
    # Processing timestamp
    data = processing_time_stamp(params.train_unknown_csv)
    # Normalizing
    data_tensor = normalize(data, is_known=False)
    # Saving tensor to .pt
    torch.save(data_tensor, params.train_unknown_pt)

    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", params.train_unknown_pt)
    print(colorama.Fore.RESET)


def generate_known():
    pass


def generate_all():
    train_csv = processing_time_stamp(params.train_csv, True)
    train_csv_normalized = normalize(train_csv, False)
    torch.save(train_csv_normalized, params.train_pt)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", params.train_pt)
    print(colorama.Fore.RESET)


def generate_verify_test():
    test_csv = processing_time_stamp(params.test_csv, False)
    print(test_csv)
    test_csv_normalized = normalize(test_csv, False)
    print(test_csv_normalized)
    torch.save(test_csv_normalized, params.test_pt)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", params.test_pt)
    print(colorama.Fore.RESET)


def generate_train_test():
    # 剪掉时间戳,train为True，test为False
    train_csv = processing_time_stamp(params.train_csv, True)
    test_csv = processing_time_stamp(params.test_csv, False)
    # print(train_csv)
    # print(test_csv)
    # 提取 target 字段
    target_columns = train_csv['target']
    train_csv_dropped = train_csv.drop(columns=['target'])

    # 拼接两个数据集
    combined_dataset = pd.concat([train_csv_dropped, test_csv], ignore_index=True)

    # 删除 uuid 和 udmap
    columns_to_delete = ['uuid', 'udmap']
    features = combined_dataset.drop(columns=columns_to_delete)
    # print(features)

    # 对特征进行归一化
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(features)

    # 将归一化后的特征重新放回数据框
    normalized_dataset = pd.DataFrame(normalized_features, columns=features.columns)

    # 分割数据集为与初始数据集数量相同的两个数据集
    num_samples = len(train_csv)
    dataset1_normalized = normalized_dataset[:num_samples]
    dataset2_normalized = normalized_dataset[num_samples:]

    # 将 target 字段拼接回去
    dataset1_normalized['target'] = target_columns
    tensor_train = torch.tensor(dataset1_normalized.values, dtype=torch.float32)
    tensor_test = torch.tensor(dataset2_normalized.values, dtype=torch.float32)

    # 保存归一化后的数据集到文件
    torch.save(tensor_train, params.train_norm_pt)
    torch.save(tensor_test, params.test_norm_pt)

    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", params.train_norm_pt)
    print("You can search your .pt at:", params.test_norm_pt)
    print(colorama.Fore.RESET)


# @TODO: To make another function to process dataset with known udmap
if __name__ == '__main__':
    generate_train_test()
