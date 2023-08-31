import colorama
import torch
import os
from torch import nn, optim
import matplotlib.pyplot as plt
from datetime import datetime

from model.IO_BinaryClassifier import BinaryClassifier
from tools.load_data import load_data

from config import Config

config = Config()

# 将数据转换为合适的形状，即 (batch_size, input_size)
data_loader = load_data(config.train_unknown_pt)

# 使用模型进行预测
# 输入特征的维度
# 创建模型实例
model = BinaryClassifier()

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=config.lr)

loss_trend = []

for epoch in range(config.num_epochs):
    print(f"Epoch {epoch + 1}/{config.num_epochs}")
    loss_epoch = []
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
        loss_epoch.append(loss.item())

        # 打印损失
    batch_loss = sum(loss_epoch) / len(loss_epoch)
    print("Batch Loss:", batch_loss)
    loss_trend.append(batch_loss)

# 创建 x 轴数据（代表迭代次数或轮数）
iterations = range(1, len(loss_trend) + 1)
# 绘制折线图
plt.plot(iterations, loss_trend, marker='o')
# 添加标题和标签
plt.title("Loss Trend Over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Loss")
# 显示网格线
plt.grid()
# 显示图形
plt.show()

current_time = datetime.now()
# 格式化时间为年月日时分
formatted_time = current_time.strftime("%Y_%m_%d_%H_%M")

model_save_path = os.path.join(config.model_save_path, "unknown_" + formatted_time + ".pkl")
plt_save_path = os.path.join(config.plt_save_path, "unknown_" + formatted_time + ".png")
torch.save(model, model_save_path)
plt.savefig(plt_save_path)

print(colorama.Fore.LIGHTGREEN_EX)
print("Training finish!!!")
print(colorama.Fore.RESET)
