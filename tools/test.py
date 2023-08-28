import torch
from torch.utils.data import Dataset, DataLoader


def F_score(raw: torch.Tensor, pred: torch.Tensor, beta: float = 1.0):
    """
    Calculate precision, recall, and F-score based on raw scores and predictions.

    Args:
        raw (torch.Tensor): The raw scores or probabilities from a model.
        pred (torch.Tensor): The binary predictions (0 or 1) from a model.
        beta (float): The beta parameter for controlling the balance between precision and recall in F-score.
                      Default is 1.0 (harmonic mean of precision and recall).

    Returns:
        float: Precision
        float: Recall
        float: F-score
    """
    TP = ((pred == 1) & (raw >= 0.5)).sum().item()
    FP = ((pred == 1) & (raw < 0.5)).sum().item()
    FN = ((pred == 0) & (raw >= 0.5)).sum().item()

    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    f_score = ((1 + beta ** 2) * precision * recall) / ((beta ** 2 * precision) + recall) if (precision + recall) > 0 else 0.0

    accuracy = (((pred == 0) & (raw < 0.5)).sum().item() + ((pred == 1) & (raw >= 0.5)).sum().item()) / len(raw)
    return precision, recall, f_score, accuracy


def test(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> (float, float, float, float):
    precision = 0
    recall = 0
    f_score = 0
    accuracy = 0
    y_pred_tensor = []
    y_raw_tensor = []

    for iterator in data_loader:
        y_pred = model(iterator["features"].to(device))
        y_pred = y_pred.squeeze()
        threshold = 0.5
        y_pred_binary = (y_pred >= threshold).float()

        # 以列表的形式将数据添加到 y_pred_tensor 和 y_raw_tensor
        y_pred_tensor.extend(y_pred_binary.tolist())
        y_raw_tensor.extend(iterator['label'].tolist())

    # 将列表转换为 PyTorch Tensor
    y_raw_tensor = torch.tensor(y_raw_tensor)
    y_pred_tensor = torch.tensor(y_pred_tensor)

    # 使用 F_score 函数进行评估，假设 F_score 函数接受两个 Tensor 作为参数
    precision, recall, f_score, accuracy = F_score(y_raw_tensor, y_pred_tensor)
    print(precision)
    print(recall)
    print(f_score)
    print(accuracy)
    return precision, recall, f_score, accuracy
