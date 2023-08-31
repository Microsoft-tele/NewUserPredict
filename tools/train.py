import torch

from tools.config_file import NewUserPredictParams
from torch.utils.data import Dataset, DataLoader

params = NewUserPredictParams()


def train(model: torch.nn.Module, optimizer: torch.optim, criterion, data_loader: DataLoader, num_epochs: int,
          device: torch.device) -> (list, float):
    """

    Args:
        model:
        optimizer:
        criterion:
        data_loader:
        num_epochs:
        device:

    Returns:

    """
    loss_trend = []
    best_loss = 100
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        loss_epoch = []
        for batch in data_loader:
            # 提取特征和标签
            features = batch['features'].to(device)
            # print(features.shape)
            labels = batch['label'].to(device)
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

        batch_loss = sum(loss_epoch) / len(loss_epoch)
        # 打印损失
        print("Batch Loss:", batch_loss)
        loss_trend.append(batch_loss)
        if batch_loss < best_loss:
            best_loss = batch_loss
    return loss_trend, best_loss
