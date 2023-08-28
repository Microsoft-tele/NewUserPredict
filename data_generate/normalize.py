import colorama
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tools import config_file
from tools.normalize import normalize_by_columns

params = config_file.NewUserPredictParams()

if __name__ == '__main__':
    df_train_processed = pd.read_csv(params.train_processed_csv)
    df_test_processed = pd.read_csv(params.test_processed_csv)
    df_train_processed = df_train_processed.drop(df_train_processed.columns[0], axis=1)
    df_test_processed = df_test_processed.drop(df_test_processed.columns[0], axis=1)

    df_train_processed = df_train_processed.drop(columns=['key1', 'key2', 'key3', 'key4', 'key5', 'key6', 'key7', 'key8', 'key9'])
    df_test_processed = df_test_processed.drop(columns=['key1', 'key2', 'key3', 'key4', 'key5', 'key6', 'key7', 'key8', 'key9'])
    normalize_by_columns(df_train_processed, df_test_processed)
