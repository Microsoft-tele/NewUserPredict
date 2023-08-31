import os
import sys

import colorama
import pandas as pd
import torch
from torch.utils.data import Subset, DataLoader

current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)

import torch.nn
from torch import nn
from model import ConfigBase
from tools.config_file import NewUserPredictParams
from tools.load_data import CustomDataset, PredictDataset
from tools.normalize import normalize_by_columns

params = NewUserPredictParams()


def load_data_from_csv(train_dataset_path: str, test_dataset_path: str, batch_size: int, is_train: int,
                       division_rate=0.8) -> DataLoader:
    """

    Args:
        batch_size:
        division_rate:
        train_dataset_path:
        test_dataset_path:
        is_train: 0 [train] 1 [verify] 2[test]

    Returns:

    """
    pd.set_option('display.max_columns', None)  # 显示所有列
    df_train = pd.read_csv(train_dataset_path)
    df_test = pd.read_csv(test_dataset_path)
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    df_combined_normalized = normalize_by_columns(df_combined, ['eid', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8',
                                                                'key1', 'key2', 'key3', 'key4', 'key5', 'key6', 'key7',
                                                                'key8', 'key9', 'date', 'hour', 'weekday']).drop(
        columns=['one_hot'])
    len_train = len(df_train)
    df_train_normalized = df_combined_normalized.iloc[:len_train, :]
    df_test_normalized = df_combined_normalized.iloc[len_train:, :].drop(columns=['target'])

    tensor_train = torch.tensor(df_train_normalized.to_numpy(), dtype=torch.float32)
    tensor_test = torch.tensor(df_test_normalized.to_numpy(), dtype=torch.float32)
    # 计算分割索引
    total_samples = len(tensor_train)
    train_size = int(division_rate * total_samples)

    res_dataloader = None
    if is_train == 0:
        dataset = CustomDataset(tensor_train)
        subset = Subset(dataset, range(train_size))
        res_dataloader = DataLoader(dataset=subset, batch_size=batch_size, shuffle=True)
    elif is_train == 1:
        dataset = CustomDataset(tensor_train)
        subset = Subset(dataset, range(train_size, total_samples))
        res_dataloader = DataLoader(dataset=subset, batch_size=batch_size, shuffle=False)
    else:
        subset = PredictDataset(tensor_test)
        res_dataloader = DataLoader(dataset=subset, batch_size=batch_size, shuffle=False)
    return res_dataloader


class BinaryClassifierEmbeddingConfig(ConfigBase):
    def __init__(self):
        super().__init__()
        self.embedding_input_dim = 9
        self.embedding_hidden_dims1 = 32
        self.embedding_output_dim = 10

        self.input_dim = self.embedding_output_dim + 12
        self.hidden_dim1 = 1024
        self.hidden_dim2 = 64
        self.output_dim = 1
        self.lr = 0.01


class BinaryClassifierEmbedding(nn.Module):
    def __init__(self, config: BinaryClassifierEmbeddingConfig):
        # Embedding layer
        super().__init__()
        self.config = config
        self.embedding_fc1 = nn.Linear(self.config.embedding_input_dim, self.config.embedding_hidden_dims1)
        self.embedding_fc2 = nn.Linear(self
                                       .config.embedding_hidden_dims1, self.config.embedding_output_dim)

        self.fc1 = nn.Linear(self.config.input_dim, self.config.hidden_dim1)
        self.fc2 = nn.Linear(self.config.hidden_dim1, self.config.hidden_dim2)
        self.fc3 = nn.Linear(self.config.hidden_dim2, self.config.output_dim)
        self.relu = nn.ReLU()  # 使用 ReLU 激活函数

    def forward(self, x):
        # 提取输入数据的前部分和后部分
        embedding_input = x[:, :self.config.embedding_input_dim]  # 前部分，用于嵌入层
        remaining_input = x[:, self.config.embedding_input_dim:]  # 后部分，将与嵌入结果拼接

        # 嵌入层的前向传播
        embedded_x = self.embedding_fc1(embedding_input)
        embedded_x = self.relu(embedded_x)
        embedded_x = self.embedding_fc2(embedded_x)

        # 将嵌入结果与后部分拼接
        combined_input = torch.cat((embedded_x, remaining_input), dim=1)

        # 全连接层的前向传播
        x = self.fc1(combined_input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x


if __name__ == "__main__":
    # config_model = BinaryClassifierEmbeddingConfig()
    # model = BinaryClassifierEmbedding(config_model)
    # print(model)
    data_loader = load_data_from_csv(train_dataset_path=params.train_processed_csv,
                                     test_dataset_path=params.test_processed_csv,
                                     batch_size=512, is_train=0)
