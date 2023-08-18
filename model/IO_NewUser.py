import torch
import torch.nn as nn
import torch.optim as optim


# 定义模型类
class BinaryClassifier(nn.Module):
    def __init__(self, dim_input):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(dim_input, 64)
        self.fc2 = nn.Linear(64, 1)  # 输出大小改为1
        self.sigmoid = nn.Sigmoid()  # 使用 sigmoid 激活函数

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))  # 使用 sigmoid 激活函数
        return x


if __name__ == '__main__':
    # 输入特征的维度
    input_size = 10

    # 创建模型实例
    model = BinaryClassifier(input_size)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 打印模型结构
    print(model)

    # 训练模型
    # 在这里添加你的训练数据和标签，然后使用循环来进行训练
