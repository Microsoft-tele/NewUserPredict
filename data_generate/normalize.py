import colorama
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from tools import config_file

params = config_file.NewUserPredictParams()


def normalize_all(pd_train: pd.DataFrame, pd_test: pd.DataFrame):
    # 提取 target 字段
    target_columns = pd_train['target']
    train_csv_dropped = pd_train.drop(columns=['target'])

    # 拼接两个数据集
    combined_dataset = pd.concat([train_csv_dropped, pd_test], ignore_index=True)

    # 对特征进行归一化
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(combined_dataset)

    # 将归一化后的特征重新放回数据框
    normalized_dataset = pd.DataFrame(normalized_features, columns=combined_dataset.columns)

    # 分割数据集为与初始数据集数量相同的两个数据集
    num_samples = len(pd_train)
    dataset1_normalized = normalized_dataset[:num_samples]
    dataset2_normalized = normalized_dataset[num_samples:]

    # 将 target 字段拼接回去
    dataset1_normalized['target'] = target_columns
    tensor_train = torch.tensor(dataset1_normalized.values, dtype=torch.float32)
    tensor_test = torch.tensor(dataset2_normalized.values, dtype=torch.float32)

    # 保存归一化后的数据集到文件
    print(tensor_train[0])
    torch.save(tensor_train, params.train_pt)
    torch.save(tensor_test, params.test_pt)

    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", params.train_pt)
    print("You can search your .pt at:", params.test_pt)
    print(colorama.Fore.RESET)


if __name__ == '__main__':
    df_train_processed = pd.read_csv(params.train_processed_csv)
    df_test_processed = pd.read_csv(params.test_processed_csv)
    df_train_processed = df_train_processed.drop(df_train_processed.columns[0], axis=1)
    df_test_processed = df_test_processed.drop(df_test_processed.columns[0], axis=1)
    normalize_all(df_train_processed, df_test_processed)
