import sys
import os
import json
import colorama
import pandas as pd

from tools.config_file import NewUserPredictParams
from tools.normalize import normalize_by_columns

params = NewUserPredictParams()


def knn_pd_deal(columns_remain: list):
    num_epoch = 0
    total_progress = 0

    # Load raw dataset
    train_csv = pd.read_csv(params.train_csv)
    test_cvs = pd.read_csv(params.test_csv)

    # process time stamp
    df_combined = pd.concat([train_csv, test_cvs], ignore_index=True)
    df_combined['common_ts'] = df_combined['common_ts'] / 31536000000 + 1970 - 2023
    for i in range(1, 10):
        df_combined = df_combined.assign(**{f'key{i}': None})

    # move "target" to the last column
    target = df_combined["target"]
    df_combined = df_combined.drop(columns=['target'])
    df_combined['target'] = target

    # stuff key1 - key9
    # for i in range(len(df_combined['udmap'])):
    #     item = df_combined['udmap'][i]
    #     dict = None
    #     try:
    #         dict = json.loads(item)
    #     except json.JSONDecodeError:
    #         dict = {
    #             "unknow": True
    #         }
    #     for j in range(1, 10):
    #         try:
    #             df_combined.at[i, f'key{j}'] = int(dict[f'key{j}'])
    #         except KeyError:
    #             df_combined.at[i, f'key{j}'] = -1
    #
    #     num_epoch += 1
    #     if num_epoch == 10000:
    #         total_progress += 1
    #         print(f"Current progress: {total_progress} w")
    #         num_epoch = 0

    # filter columns that are included in columns_remain
    filtered_columns = [col for col in df_combined.columns if any(word in col for word in columns_remain)]
    df_remain = df_combined[filtered_columns]

    # normalize the df_remain
    columns_remain.pop(2)

    # print(columns_remain)
    df_normalized = normalize_by_columns(df_remain, columns_remain[1: -1])

    df_train_normalized = df_normalized.iloc[:len(train_csv), :]
    df_test_normalized = df_normalized.iloc[len(train_csv):, :]

    print(df_train_normalized.columns)
    print("------------------")
    print(df_train_normalized)
    print("==================")
    print(df_test_normalized.columns)
    print("------------------")
    print(df_test_normalized)

    df_train_normalized.to_csv(params.train_knn_csv, index=False)
    df_test_normalized.to_csv(params.test_knn_csv, index=False)
    print("Save successfully!")


def load_knn_data(dataset: pd.DataFrame, is_train: bool):
    pass


if __name__ == "__main__":
    # columns_remain = ['uuid', 'eid', 'common_ts', 'x2', 'x4', 'x5', 'x6', 'x7', 'x8', 'key2', 'key3', 'key4', 'key5',
    #                   'target']
    columns_remain = ['uuid', 'eid', 'common_ts', 'x2', 'x5', 'x6', 'x7', 'target']
    knn_pd_deal(columns_remain)
