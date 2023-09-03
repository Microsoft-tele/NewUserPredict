import sys

import numpy as np

from tools.config_file import NewUserPredictParams

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
    data['common_ts_dt'] = pd.to_datetime(data['common_ts'], unit='ms')
    # 删除原common_ts列data.drop(columns=['common_ts'], inplace=True)
    data['common_ts'] = data['common_ts'] / 31536000000 + 1970 - 2023
    # 提取日期和小时
    data['date'] = data['common_ts_dt'].dt.day
    data['hour'] = data['common_ts_dt'].dt.hour
    data['weekday'] = data['common_ts_dt'].dt.weekday  # 0=Monday, 6=Sunday
    # @TODO: 注意回来看这里的时间戳处理
    data['sin_norm'] = np.sin(2 * np.pi * (data['common_ts'] - 0.567872) / 0.008717)
    data['cos_norm'] = np.cos(2 * np.pi * (data['common_ts'] - 0.567872) / 0.008717)
    data['sin'] = np.sin(2 * np.pi * data['common_ts'])
    data['cos'] = np.cos(2 * np.pi * data['common_ts'])

    # 保存处理后的数据集到新文件
    # data.to_csv('../dataset/train_unknown_DayHour.csv', index=False)
    return data


def processing_eid(df_dataset: pd.DataFrame) -> pd.DataFrame:
    # 按 'eid' 分组并计算 'target' 的平均值
    eid_target_mean = df_dataset['target'].groupby(df_dataset['eid']).mean()

    # 获取 'eid' 和相应的目标值
    eid = eid_target_mean.index.values
    target = eid_target_mean.values

    # 创建一个新的 DataFrame
    eid_target = pd.DataFrame({"eid": eid, "eid_target": target})

    # 通过 'eid' 合并 total_df 和 eid_target
    df_dataset = pd.merge(df_dataset, eid_target, on="eid", how="left")
    # 显示前几行的数据
    return df_dataset


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


def fill_key_value(df_dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Filling the key1 to key9 from json udmap
    Args:
        df_dataset:

    Returns:

    """
    df_combined = df_dataset
    num_epoch = 0
    total_progress = 0

    for i in range(1, 10):
        df_combined = df_combined.assign(**{f'key{i}': None})
    df_target = df_combined.pop('target')
    df_combined['target'] = df_target

    for i in range(len(df_combined['udmap'])):
        item = df_combined['udmap'][i]
        json_dict = None
        try:
            json_dict = json.loads(item)
        except json.JSONDecodeError:
            json_dict = {
                "unknown": True
            }
        for j in range(1, 10):
            try:
                df_combined.at[i, f'key{j}'] = int(json_dict[f'key{j}'])
            except KeyError:
                df_combined.at[i, f'key{j}'] = -1
        num_epoch += 1
        if num_epoch == 10000:
            total_progress += 1
            print(f"Current progress: {total_progress} w")
            num_epoch = 0

    return df_combined
