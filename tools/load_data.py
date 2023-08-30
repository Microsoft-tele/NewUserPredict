import torch
from torch.utils.data import Dataset, DataLoader, Subset
from tools.config_file import NewUserPredictParams

params = NewUserPredictParams()


class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        sample = {
            'uuid': self.data_tensor[idx, 0],  # uuid which is used to tag every row
            'features': self.data_tensor[idx, 1:-1],  # 特征，排除第一个和最后一个元素
            'label': self.data_tensor[idx, -1]  # 标签
        }
        return sample


class PredictDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        return self.data_tensor[idx]


def load_data(pt_file_path: str, batch_size: int, division_rate: float, is_train: bool = True) -> DataLoader:
    """
    Loading dataset from .pt file which has been previously processed by dada_generated factor

    Args:
        division_rate:
        batch_size:
        pt_file_path: pt file path
        is_train: True: load train dataset, else load test dataset

    Returns:

    """
    data_tensor = torch.load(pt_file_path)

    # 创建数据集和数据加载器
    dataset = CustomDataset(data_tensor)

    # 计算分割索引
    total_samples = len(data_tensor)
    train_size = int(division_rate * total_samples)

    train_subset = Subset(dataset, range(train_size))
    test_subset = Subset(dataset, range(train_size, total_samples))

    # 创建数据加载器
    if is_train:
        return DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(test_subset, batch_size=batch_size, shuffle=False)  # 不需要在测试时打乱顺序


def load_test():
    data_tensor = torch.load(params.test_pt)
    # 创建数据集和数据加载器
    dataset = PredictDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=params.batch_size, shuffle=False)
    return loader


if __name__ == "__main__":
    # 获取第一个批次
    dataloader = load_test()
    print(len(dataloader))
    first_batch = next(iter(dataloader))

    print("shape of first batch:")
    print(first_batch.shape)  # 打印特征
    print(first_batch[0])  # 打印特征
