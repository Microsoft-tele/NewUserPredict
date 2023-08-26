import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from tools import config_file
from datetime import datetime

params = config_file.NewUserPredictParams()


def processing_time_stamp(file_path: str, is_true=bool):
    """
    Through this function, timestamp will be divided into date, hour and weekday
    Please make sure that dataset owns the column whose name is common_ts
    Dataset is loaded by config file, so you have no need to concern about dataset

    :notice: This function could be used firstly, no matter unknown or known
    :author: Micros0ft
    :return: data: pd
    """
    data = pd.read_csv(file_path)

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

    # if is_true:
    #     target = data.columns[-4]
    #     target_data = data[target]
    #     data.drop(columns=[target], inplace=True)
    #     data[target] = target_data
    # else:
    #     pass

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


def binary_list_to_num(li: list) -> int:
    binary_str = ''.join(map(str, li))
    decimal_num = int(binary_str, 2)
    return decimal_num


def one_hot():
    one_hot_row = [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    df = pd.read_csv('./train_known_processed.csv')
    for i, row in df.iterrows():
        for j in range(1, 10):
            if row[f'key{j}'] == -1:
                one_hot_row[j - 1] = 0

            else:
                one_hot_row[j - 1] = 1
        num = binary_list_to_num(one_hot_row)
        # 将整数赋值给 'one_hot' 列
        df.at[i, 'one_hot'] = num
    df.to_csv('./train_known_processed_one_hot.csv')
    print(df)


if __name__ == "__main__":
    data = processing_time_stamp()
    data_tensor = normalize(data, is_known=False)
    torch.save(data_tensor, params.train_unknown_pt)
