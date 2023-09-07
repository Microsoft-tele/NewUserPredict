import sys

import colorama
import pandas as pd

from tools.config_file import NewUserPredictParams

params = NewUserPredictParams()

from tools import *
from tools.normalize import normalize_by_columns, standardize_by_columns


def generate_train_test():
    # 剪掉时间戳,train为True，test为False
    train_csv = processing_time_stamp(params.train_csv)
    test_csv = processing_time_stamp(params.test_csv)
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
    pass
