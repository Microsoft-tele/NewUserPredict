import torch
import torch.nn as nn
import torch.optim as optim

from config import Config

config = Config()


# 定义模型类
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(config.input_size, config.hide_size)
        self.fc2 = nn.Linear(config.hide_size, 1)  # 输出大小改为1
        self.sigmoid = nn.Sigmoid()  # 使用 sigmoid 激活函数

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))  # 使用 sigmoid 激活函数
        return x


if __name__ == '__main__':
    # 创建模型实例
    model = BinaryClassifier()

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 打印模型结构
    print(model)

    # 训练模型
    # 在这里添加你的训练数据和标签，然后使用循环来进行训练
