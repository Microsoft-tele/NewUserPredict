import os
import sys

import colorama
import torch

current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim

from tools.config_file import NewUserPredictParams
from model import ConfigBase

params = NewUserPredictParams()


class BinaryClassifierConfig(ConfigBase):
    def __init__(self):
        super().__init__()
        self.input_dim = 14
        self.hidden_dim1 = 64
        self.hidden_dim2 = 64
        self.output_dim = 1

        self.batch_size = 512
        self.lr = 0.01
        self.epoch_num = 1200


# 定义模型类
class BinaryClassifier(nn.Module):
    def __init__(self, config: BinaryClassifierConfig):
        super(BinaryClassifier, self).__init__()
        self.config = config
        self.fc1 = nn.Linear(self.config.input_dim, self.config.hidden_dim1)
        self.fc2 = nn.Linear(self.config.hidden_dim1, self.config.hidden_dim2)  # 新添加的隐藏层
        self.fc3 = nn.Linear(self.config.hidden_dim1, self.config.output_dim)
        self.relu = nn.ReLU()  # 使用 ReLU 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))  # 新隐藏层的激活函数也可以是 ReLU
        x = self.fc3(x)
        x = torch.sigmoid(x)  # 将 sigmoid 激活函数添加到最后
        return x


if __name__ == '__main__':
    # 创建模型实例
    config_model = BinaryClassifierConfig()
    model = BinaryClassifier(config_model)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 打印模型结构
    print(model)

    # 训练模型
    # 在这里添加你的训练数据和标签，然后使用循环来进行训练
