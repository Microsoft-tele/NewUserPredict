import math

import pandas as pd
import torch
import colorama
import json
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F


from tools.config_file import NewUserPredictParams
from dataset import processing_time_stamp, normalize, binary_list_to_num

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


def convert_udmap(udmap_str):
    try:
        return json.loads(udmap_str)
    except json.JSONDecodeError:
        return {'unknown': True}


def generate_known(is_true=bool):

    if is_true:
        df = pd.read_csv(params.train_csv)
    else:
        df = pd.read_csv(params.test_csv)

    # 使用 apply 一次性转换 udmap 列
    df['udmap'] = df['udmap'].apply(convert_udmap)

    # 一次性添加空列
    for i in range(1, 10):
        df = df.assign(**{f'key{i}': None})
    df['one_hot'] = None

    # 移动 target 列到最后
    if is_true:
        df = df[[col for col in df.columns if col != 'target'] + ['target']]
    else:
        pass

    # 填充 key 列的值
    for i, row in df.iterrows():
        for j in range(1, 10):
            try:
                df.at[i, f'key{j}'] = row['udmap'][f'key{j}']
            except KeyError:
                df.at[i, f'key{j}'] = -1

    try:
        df_df = pd.DataFrame(df)
        if is_true:
            df.to_csv('./train_known_processed.csv')
        else:
            df.to_csv('./test_known_processed.csv')
    except:
        print('error')


def one_hot():
    one_hot_row = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    df_train = pd.read_csv('./train_known_processed.csv')
    df_test = pd.read_csv('./test_known_processed.csv')

    for i, row in df_train.iterrows():
        for j in range(1, 10):
            if row[f'key{j}'] == -1:
                one_hot_row[j - 1] = 0

            else:
                one_hot_row[j - 1] = 1
        num = binary_list_to_num(one_hot_row)
        # 将整数赋值给 'one_hot' 列
        df_train.at[i, 'one_hot'] = num
    df_train.to_csv('./train_known_processed_one_hot.csv')
    # print(df_train)

    for i, row in df_test.iterrows():
        for j in range(1, 10):
            if row[f'key{j}'] == -1:
                one_hot_row[j - 1] = 0

            else:
                one_hot_row[j - 1] = 1
        num = binary_list_to_num(one_hot_row)
        # 将整数赋值给 'one_hot' 列
        df_test.at[i, 'one_hot'] = num
    df_test.to_csv('./test_known_processed_one_hot.csv')
    # print(df_test)


def adjust_one_hot_csv():

    # 剪掉时间戳
    train_one_hot_csv = processing_time_stamp(params.train_known_processed_one_hot_csv, True)
    test_one_hot_csv = processing_time_stamp(params.test_known_processed_one_hot_csv, False)

    # 调整 target 字段到最后一列
    train_csv_adjust = train_one_hot_csv.drop(columns=['target'])
    train_csv_adjust['target'] = train_one_hot_csv['target']

    # 删除第一列与第二列数据以及udmap列
    train_csv_dropped = train_csv_adjust.drop(columns=[train_csv_adjust.columns[0], train_csv_adjust.columns[1], train_csv_adjust.columns[2], train_csv_adjust.columns[4]])
    print(train_csv_dropped)
    print(train_csv_dropped.columns)

    test_csv_dropped = test_one_hot_csv.drop(
        columns=[test_one_hot_csv.columns[0], test_one_hot_csv.columns[1], test_one_hot_csv.columns[2],
                 test_one_hot_csv.columns[4]])
    print(test_csv_dropped)
    print(test_csv_dropped.columns)






def generate_all():
    train_csv = processing_time_stamp(params.train_csv, True)
    train_csv_normalized = normalize(train_csv, False)
    torch.save(train_csv_normalized, params.train_all_pt)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", params.train_all_pt)
    print(colorama.Fore.RESET)


def generate_verify_test():
    test_csv = processing_time_stamp(params.test_csv, False)
    print(test_csv)
    test_csv_normalized = normalize(test_csv, False)
    print(test_csv_normalized)
    torch.save(test_csv_normalized, params.test_all_pt)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", params.test_all_pt)
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
    adjust_one_hot_csv()