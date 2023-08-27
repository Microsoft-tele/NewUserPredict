import sys

import colorama

from tools.config_file import NewUserPredictParams

params = NewUserPredictParams()

from tools import *


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


def generate_all():
    df_train = pd.read_csv(params.train_csv)
    df_test = pd.read_csv(params.test_csv)

    # 使用 apply 一次性转换 udmap 列
    df_train['udmap'] = df_train['udmap'].apply(convert_udmap)
    df_test['udmap'] = df_test['udmap'].apply(convert_udmap)

    # 一次性添加空列
    for i in range(1, 10):
        df_train = df_train.assign(**{f'key{i}': None})
    df_train['one_hot'] = None
    for i in range(1, 10):
        df_test = df_test.assign(**{f'key{i}': None})
    df_test['one_hot'] = None

    # 移动 target 列到最后
    df_train = df_train[[col for col in df_train.columns if col != 'target'] + ['target']]

    # 填充 key 列的值
    for i, row in df_train.iterrows():
        for j in range(1, 10):
            try:
                df_train.at[i, f'key{j}'] = row['udmap'][f'key{j}']
            except KeyError:
                df_train.at[i, f'key{j}'] = -1

    for i, row in df_test.iterrows():
        for j in range(1, 10):
            try:
                df_test.at[i, f'key{j}'] = row['udmap'][f'key{j}']
            except KeyError:
                df_test.at[i, f'key{j}'] = -1

    df_train, df_test = one_hot(df_train, df_test)
    df_adjusted_train, df_adjusted_test = adjust_one_hot_csv(df_train, df_test)

    # standard_train, standard_test = standard_csv(df_adjusted_train, df_adjusted_test)
    # print(standard_train)
    #
    # standard_train.to_csv(params.train_processed_csv)
    # standard_test.to_csv(params.test_processed_csv)

    # tensor_train = torch.tensor(df_adjusted_train.values, dtype=torch.float32)
    # tensor_test = torch.tensor(df_adjusted_test.values, dtype=torch.float32)
    #
    # torch.save(tensor_train, params.train_pt)
    # torch.save(tensor_test, params.test_pt)

    df_adjusted_train.to_csv(params.train_processed_csv, index=False)
    df_adjusted_test.to_csv(params.test_processed_csv, index=False)

    print(colorama.Fore.LIGHTGREEN_EX)
    print("You can find final processed train dataset at : ", params.train_processed_csv)
    print("You can find final processed test dataset at : ", params.test_processed_csv)
    print(colorama.Fore.RESET)


def generate_verify_test():
    test_csv = processing_time_stamp(params.test_csv)
    print(test_csv)
    test_csv_normalized = normalize(test_csv, False)
    print(test_csv_normalized)
    torch.save(test_csv_normalized, params.test_pt)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", params.test_pt)
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
    generate_all()
