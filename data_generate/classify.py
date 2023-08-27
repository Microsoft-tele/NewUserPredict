import os
import sys

current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)
import pandas as pd
from tools.config_file import NewUserPredictParams

params = NewUserPredictParams()


def divide(df_csv: pd.DataFrame, eids: list) -> pd.DataFrame:
    """
    :author Micros0ft
    :date 2023/8/27
    :param df_csv: Processed training dataset, udmap has been divided into 9 different keys
    :param eids: List of eids for dividing data series into separate pd.DataFrames
    :return: train_df divided according to eids, test_df divided according to eids
    """
    train_df_list = []  # Store divided training DataFrames based on eids

    # For each eid in eids, extract corresponding data from training and testing DataFrames
    for eid in eids:
        train_df_eid = df_csv[df_csv['eid'] == eid]

        train_df_list.append(train_df_eid)

    # Concatenate all divided DataFrames into single DataFrames
    df_train = pd.concat(train_df_list, ignore_index=True)

    return df_train


if __name__ == '__main__':
    key2_key3 = [26, 40, 3, 38, 25, 12, 7, 0, 27, 34, ]
    key4_key5 = [2, 5, ]
    unknown = [41, 36, 31, 30, 4, 1, 19, 13, 15, 20, 10, 9, 29, 37, 32, 21, 39, 35, 11, 8, 33, 42, 28, 14, 16, 23, 6, 22, 18, 17, 24, ]
    columns_1 = ['key1', 'key4', 'key5', 'key6', 'key7', 'key8', 'key9']
    columns_2 = ['key6', 'key7', 'key8', 'key9']
    columns_3 = ['key1', 'key2', 'key3', 'key4', 'key5', 'key6', 'key7', 'key8', 'key9']
    train_processed_csv = pd.read_csv(params.train_processed_csv)
    test_processed_csv = pd.read_csv(params.test_processed_csv)
    dataset = [train_processed_csv, test_processed_csv]
    train_df_list = []
    test_df_list = []
    is_train = True

    for df in dataset:
        key2_key3_df = divide(df, key2_key3)
        key4_key5_df = divide(df, key4_key5)
        unknown_df = divide(df, unknown)
        if is_train:
            train_df_list.append(key2_key3_df.drop(columns=columns_1))
            train_df_list.append(key4_key5_df.drop(columns=columns_2))
            train_df_list.append(unknown_df.drop(columns=columns_3))
            is_train = False
        else:
            test_df_list.append(key2_key3_df.drop(columns=columns_1))
            test_df_list.append(key4_key5_df.drop(columns=columns_2))
            test_df_list.append(unknown_df.drop(columns=columns_3))
    combined_df_list = []
    for i in range(len(train_df_list)):
        combined_df_list.append(pd.concat([train_df_list[i], test_df_list[i]], ignore_index=True))



    # key2_key3_df.to_csv('./1.csv')
    # key4_key5_df.to_csv('./2.csv')
    # unknown_df.to_csv('./3.csv')


