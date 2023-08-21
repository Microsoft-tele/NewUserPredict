import torch
from torch import nn, optim

from model.IO_NewUser import BinaryClassifier
from tools.load_data import load_data

from config import Config

config = Config()

# 将数据转换为合适的形状，即 (batch_size, input_size)
data_loader = load_data(config.train_unknown_pt)

# 使用模型进行预测
# 输入特征的维度
# 创建模型实例
model = BinaryClassifier(config.input_size)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=config.lr)

for epoch in range(config.num_epochs):
    print(f"Epoch {epoch + 1}/{config.num_epochs}")
    for batch in data_loader:
        # 提取特征和标签
        features = batch['features']
        # print(features.shape)
        labels = batch['label']
        labels = labels.unsqueeze(1)

        # 清零梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(features)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印损失
    print("Batch Loss:", loss.item())
