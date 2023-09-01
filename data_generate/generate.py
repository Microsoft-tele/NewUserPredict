import sys

import colorama
import pandas as pd
import torch

from tools.config_file import NewUserPredictParams

params = NewUserPredictParams()

from tools import *
from tools.normalize import normalize_by_columns, standardize_by_columns


# def generate_unknown():
#     # Processing timestamp
#     data = processing_time_stamp(params.train_unknown_csv)
#     # Normalizing
#     data_tensor = normalize(data, is_known=False)
#     # Saving tensor to .pt
#     torch.save(data_tensor, params.train_unknown_pt)
#
#     print(colorama.Fore.LIGHTGREEN_EX)
#     print("Convert dataset successfully!!!")
#     print("You can search your .pt at:", params.train_unknown_pt)
#     print(colorama.Fore.RESET)


def feature_engineering(normalize_or_standardize: bool):
    """

    Args:
        normalize_or_standardize: True -> normalize else standardize

    Returns:

    """
    pd.set_option('display.max_columns', None)
    df_train = pd.read_csv(params.train_csv)
    df_test = pd.read_csv(params.test_csv)
    # Concat train and test
    df_combined = pd.concat([df_train, df_test], ignore_index=True)

    # fill key according to udmap
    df_combined = fill_key_value(df_combined)

    # handle timestamp
    df_combined = processing_time_stamp(df_combined)
    df_combined = processing_eid(df_combined)

    # select normalize or standardize
    # ['eid', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
    #  'key1', 'key2', 'key3', 'key4', 'key5', 'key6', 'key7',
    #  'key8', 'key9', 'date', 'hour', 'weekday']
    if normalize_or_standardize:
        df_combined = normalize_by_columns(df_combined, ['eid', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
                                                         'key1', 'key2', 'key3', 'key4', 'key5', 'key6', 'key7',
                                                         'key8', 'key9', 'date', 'hour', 'weekday'])
    else:
        df_combined = standardize_by_columns(df_combined, ['eid', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
                                                           'key1', 'key2', 'key3', 'key4', 'key5', 'key6', 'key7',
                                                           'key8', 'key9', 'date', 'hour', 'weekday'])

    df_train_processed: pd.DataFrame = df_combined.iloc[:len(df_train), :]
    df_test_processed: pd.DataFrame = df_combined.iloc[len(df_train):, :]

    df_train_processed.to_csv(params.train_processed_csv, index=False)
    df_test_processed.to_csv(params.test_processed_csv, index=False)

    print(colorama.Fore.LIGHTGREEN_EX)
    print("You can find final processed train dataset at : ", params.train_processed_csv)
    print("You can find final processed test dataset at : ", params.test_processed_csv)
    print(colorama.Fore.RESET)


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
    # tensor_train = torch.load(params.train_pt)
    # print(tensor_train.shape)
    # print(tensor_train[0])
    feature_engineering(normalize_or_standardize=True)
