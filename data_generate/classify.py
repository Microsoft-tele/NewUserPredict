import os
import sys

import colorama
import torch

from tools.normalize import normalize_by_columns

current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)

import pandas as pd
from tools.config_file import NewUserPredictParams

params = NewUserPredictParams()


def divide_by_eids(df_csv: pd.DataFrame, eids: list) -> pd.DataFrame:
    """
    :author Micros0ft
    :date 2023/8/27
    :param df_csv: Processed training dataset, udmap has been divided into 9 different keys
    :param eids: List of eids for dividing data series into separate pd.DataFrames
    :return: train_df divided according to eids, test_df divided according to eids
    """
    df_list = []  # Store divided training DataFrames based on eids

    # For each eid in eids, extract corresponding data from training and testing DataFrames
    for eid in eids:
        train_df_eid = df_csv[df_csv['eid'] == eid]

        df_list.append(train_df_eid)

    # Concatenate all divided DataFrames into single DataFrames
    df_train = pd.concat(df_list, ignore_index=True)

    return df_train


import pandas as pd


def divide_by_one_hot(df_csv: pd.DataFrame, one_hot: int) -> pd.DataFrame:
    """
    Extract rows from the DataFrame where the value in the 'one_hot' column matches the provided one_hot value.

    Args:
        df_csv (pd.DataFrame): Processed training dataset.
        one_hot (int): Value to filter the 'one_hot' column.

    Returns:
        pd.DataFrame: DataFrame with rows that match the provided one_hot value.
    """
    filtered_df = df_csv[df_csv['one_hot'] == one_hot]
    return filtered_df


if __name__ == '__main__':
    key2_key3 = [26, 40, 3, 38, 25, 12, 7]
    key3 = [0, 27, 34]
    key4_key5 = [2, 5, ]

    # If one_hot == 0 单独一组
    unknown = [41, 36, 31, 30, 4, 1, 19, 13, 15, 20, 10, 9, 29, 37, 32, 21, 39, 35, 11, 8, 33, 42, 28, 14, 16, 23, 6,
               22, 18, 17, 24, ]

    # the columns will be deleted
    columns_1 = ['key1', 'key4', 'key5', 'key6', 'key7', 'key8', 'key9']
    columns_2 = ['key1', 'key2', 'key4', 'key5', 'key6', 'key7', 'key8', 'key9']
    columns_3 = ['key6', 'key7', 'key8', 'key9']
    columns_4 = ['key1', 'key2', 'key3', 'key4', 'key5', 'key6', 'key7', 'key8', 'key9']
    train_processed_csv = pd.read_csv(params.train_processed_csv)
    test_processed_csv = pd.read_csv(params.test_processed_csv)
    dataset = [train_processed_csv, test_processed_csv]

    train_df_list = []
    len_of_every_train_dataset = []
    test_df_list = []
    is_train = True

    for df in dataset:

        # indices_to_remove = unknown_df.index
        # df_cleaned = df.drop(indices_to_remove)

        key2_key3_df = divide_by_eids(df, key2_key3)
        key3_df = divide_by_eids(df, key3)
        key4_key5_df = divide_by_eids(df, key4_key5)
        unknown_df = divide_by_eids(df, unknown)

        # 将满足条件的行从 key2_key3_df 移动到 unknown_df 下面
        condition = key2_key3_df['one_hot'] == 0
        rows_to_move = key2_key3_df[condition]

        # 从 key2_key3_df 中移除这些行
        key2_key3_df = key2_key3_df[~condition]

        # 将这些行添加到 unknown_df 下面
        unknown_df = pd.concat([unknown_df, rows_to_move], ignore_index=True)

        if is_train:
            train_df_list.append(key2_key3_df.drop(columns=columns_1))
            len_of_every_train_dataset.append(len(key2_key3_df))
            train_df_list.append(key3_df.drop(columns=columns_2))
            len_of_every_train_dataset.append(len(key3_df))
            train_df_list.append(key4_key5_df.drop(columns=columns_3))
            len_of_every_train_dataset.append(len(key4_key5_df))
            train_df_list.append(unknown_df.drop(columns=columns_4))
            len_of_every_train_dataset.append(len(unknown_df))
            is_train = False
        else:
            test_df_list.append(key2_key3_df.drop(columns=columns_1))
            test_df_list.append(key3_df.drop(columns=columns_2))
            test_df_list.append(key4_key5_df.drop(columns=columns_3))
            test_df_list.append(unknown_df.drop(columns=columns_4))

    combined_df_list = []

    for i in range(len(train_df_list)):
        combined_df_list.append(pd.concat([train_df_list[i], test_df_list[i]], ignore_index=True))

    for i in range(len(combined_df_list)):
        # Notice: Before saving, we have not divided dataset into train and test
        combined_df_list[i].to_csv(params.train_classified_csv[i])

    normalization_columns = [
        ['eid', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'key2', 'key3', 'date', 'hour', 'weekday'],
        ['eid', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'key3', 'date', 'hour', 'weekday'],
        ['eid', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'key1', 'key2', 'key3', 'key4', 'key5', 'date', 'hour',
         'weekday'],
        ['eid', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'date', 'hour', 'weekday'],
    ]
    train_tensors = []
    test_tensors = []
    for i in range(len(combined_df_list)):
        normalized_df = normalize_by_columns(combined_df_list[i], normalization_columns[i]).drop(columns=['one_hot'])
        # print(colorama.Fore.LIGHTGREEN_EX)
        # print(normalized_df.columns)
        # print(colorama.Fore.RESET)

        train_normalized_df = normalized_df.iloc[:len_of_every_train_dataset[i], :]
        test_normalized_df = normalized_df.iloc[len_of_every_train_dataset[i]:, :].drop(columns=['target'])

        # normalized_dataset.to_csv(f"./{i}_normalization.csv")
        train_normalized_np = train_normalized_df.to_numpy()
        test_normalized_np = test_normalized_df.to_numpy()
        train_tensors.append(torch.tensor(train_normalized_np, dtype=torch.float32))
        test_tensors.append(torch.tensor(test_normalized_np, dtype=torch.float32))
        print(train_tensors[i].shape)
        print(test_tensors[i].shape)

    for i in range(len(train_tensors)):
        torch.save(train_tensors[i], params.train_classified_pt[i])
        torch.save(test_tensors[i], params.test_classified_pt[i])
        print(colorama.Fore.LIGHTGREEN_EX)
        print("You can find your train .pt file at:", params.train_classified_pt[i])
        print("You can find your test .pt file at:", params.test_classified_pt[i])
