import torch
import torch.nn as nn
import torch.optim as optim

from tools.config_file import NewUserPredictParams

params = NewUserPredictParams()


# 定义模型类
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(params.input_size, params.hidden_size1)
        self.fc2 = nn.Linear(params.hidden_size1, params.hidden_size2)  # 新添加的隐藏层
        self.fc3 = nn.Linear(params.hidden_size2, 1)  # 输出大小改为1
        self.relu = nn.ReLU()  # 使用 ReLU 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))  # 新隐藏层的激活函数也可以是 ReLU
        x = self.fc3(x)
        x = torch.sigmoid(x)  # 将 sigmoid 激活函数添加到最后
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
