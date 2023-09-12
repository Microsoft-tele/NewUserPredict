import os
import sys

import colorama

current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)
import pandas as pd
from tools.config_file import NewUserPredictParams
from tools.normalize import normalize_by_columns, standardize_by_columns
from tools.statistic import statistic_certain_column
from tools import fill_key_value, processing_time_stamp, processing_eid, adding_frequency, processing_xith

params = NewUserPredictParams()


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
    # df_combined = fill_key_value(df_combined)

    # handle timestamp
    df_combined = processing_time_stamp(df_combined)
    df_combined = processing_eid(df_combined)
    df_combined = processing_xith(df_combined)

    # add frequency of x1-x8
    df_combined = adding_frequency(df_combined)

    # select normalize or standardize
    # ['eid', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
    #  'key1', 'key2', 'key3', 'key4', 'key5', 'key6', 'key7',
    #  'key8', 'key9', 'date', 'hour', 'weekday']
    features_list = ['eid']
    if normalize_or_standardize:
        df_combined = normalize_by_columns(df_combined, features_list)
    else:
        df_combined = standardize_by_columns(df_combined, features_list)

    df_train_processed: pd.DataFrame = df_combined.iloc[:len(df_train), :]
    df_test_processed: pd.DataFrame = df_combined.iloc[len(df_train):, :]

    df_train_processed.to_csv(params.train_processed_csv, index=False)
    df_test_processed.to_csv(params.test_processed_csv, index=False)

    print(colorama.Fore.LIGHTGREEN_EX)
    print("You can find final processed train dataset at : ", params.train_processed_csv)
    print("You can find final processed test dataset at : ", params.test_processed_csv)
    print(colorama.Fore.RESET)


if __name__ == "__main__":
    feature_engineering(normalize_or_standardize=False)
