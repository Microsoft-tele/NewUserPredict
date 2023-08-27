import sys
from tools.config_file import NewUserPredictParams

import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from . import *
import json
import pandas as pd

params = NewUserPredictParams()


def processing_time_stamp(df: pd.DataFrame):
    """
    Through this function, timestamp will be divided into date, hour and weekday
    Please make sure that dataset owns the column whose name is common_ts
    Dataset is loaded by config file, so you have no need to concern about dataset

    :notice: This function could be used firstly, no matter unknown or known
    :author: Micros0ft
    :return: data: pd
    """
    data = df

    # 将时间戳列转换为日期和时间格式
    data['common_ts'] = pd.to_datetime(data['common_ts'], unit='ms')

    # 将时间戳列转换为日期和时间格式
    data['common_ts'] = pd.to_datetime(data['common_ts'], unit='ms')

    # 提取日期和小时
    data['date'] = data['common_ts'].dt.day
    data['hour'] = data['common_ts'].dt.hour
    data['weekday'] = data['common_ts'].dt.weekday  # 0=Monday, 6=Sunday

    # 删除原common_ts列data.drop(columns=['common_ts'], inplace=True)
    data.drop(columns=['common_ts'], inplace=True)

    # 保存处理后的数据集到新文件
    # data.to_csv('../dataset/train_unknown_DayHour.csv', index=False)
    return data


def normalize(data_processed_by_timestamp: pd.DataFrame, is_known: False):
    """
    To normalize dataset which has passed timestamp procession
    Finally, all columns will be normalized to scale between 0 and 1

    :author: Micros0ft
    :param data_processed_by_timestamp:
    :param is_known: To mark the type of dataset is converted in to function
    :return: data_tensor: torch.tensor
    """
    if is_known:
        print("Now we have no need to dealt with it")
    else:
        data_without_uuid_udmap = data_processed_by_timestamp.drop(data_processed_by_timestamp.columns[[0, 2]], axis=1)

        data_numpy = data_without_uuid_udmap.values

        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(data_numpy)

        # data_pd = pd.DataFrame(normalized_data)
        # print(data_pd)
        # data_tensor = torch.from_dlpack()
        data_tensor = torch.from_numpy(normalized_data).float()
        return data_tensor


def convert_udmap(udmap_str):
    try:
        return json.loads(udmap_str)
    except json.JSONDecodeError:
        return {'unknown': True}


def binary_list_to_num(li: list) -> int:
    binary_str = ''.join(map(str, li))
    decimal_num = int(binary_str, 2)
    return decimal_num


def one_hot(df_train: pd.DataFrame, df_test: pd.DataFrame):
    one_hot_row = [-1, -1, -1, -1, -1, -1, -1, -1, -1]

    for i, row in df_train.iterrows():
        for j in range(1, 10):
            if row[f'key{j}'] == -1:
                one_hot_row[j - 1] = 0

            else:
                one_hot_row[j - 1] = 1
        num = binary_list_to_num(one_hot_row)
        # 将整数赋值给 'one_hot' 列
        df_train.at[i, 'one_hot'] = num
    # df_train.to_csv(params.train_processed_csv)

    for i, row in df_test.iterrows():
        for j in range(1, 10):
            if row[f'key{j}'] == -1:
                one_hot_row[j - 1] = 0

            else:
                one_hot_row[j - 1] = 1
        num = binary_list_to_num(one_hot_row)
        # 将整数赋值给 'one_hot' 列
        df_test.at[i, 'one_hot'] = num
    # df_test.to_csv(params.test_processed_csv)
    return df_train, df_test


def adjust_one_hot_csv(df_train: pd.DataFrame, df_test: pd.DataFrame):
    # 剪掉时间戳
    train_one_hot_csv = processing_time_stamp(df_train)
    test_one_hot_csv = processing_time_stamp(df_test)

    # 调整 target 字段到最后一列
    train_csv_adjust = train_one_hot_csv.drop(columns=['target'])
    train_csv_adjust['target'] = train_one_hot_csv['target']

    # 删除第一列与第二列数据以及udmap列
    train_csv_dropped = train_csv_adjust.drop(
        columns=[train_csv_adjust.columns[2]])
    print(train_csv_dropped)
    print(train_csv_dropped.columns)

    test_csv_dropped = test_one_hot_csv.drop(
        columns=[test_one_hot_csv.columns[2]])
    return train_csv_dropped, test_csv_dropped


def standard_csv(df_train: pd.DataFrame, df_test: pd.DataFrame):
    # drop 'target' tag
    target_columns = df_train['target']
    df_train_dropped = df_train.drop(columns=['target'])

    # combine two dataset
    combined_dataset = pd.concat([df_train_dropped, df_test], ignore_index=True)

    # standard
    scaler = StandardScaler()
    df_combined_standard = scaler.fit_transform(combined_dataset)

    # restore DataFrame
    df_combined_standard = pd.DataFrame(df_combined_standard, columns=combined_dataset.columns)

    # divide dataset
    num_samples = len(df_train)
    df_train_standard = df_combined_standard[:num_samples].copy()
    df_test_standard = df_combined_standard[num_samples:].copy()

    # restore 'target' tag
    df_train_standard['target'] = target_columns

    return df_train_standard, df_test_standard

