import torch
from torch.utils.data import Dataset, DataLoader
from config import Config

config = Config()


class CustomDataset(Dataset):
    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        sample = {
            'features': self.data_tensor[idx, :-1],  # 特征
            'label': self.data_tensor[idx, -1]  # 标签
        }
        return sample


def load_data(filepath: str):
    data_tensor = torch.load(filepath)

    # 创建数据集和数据加载器
    dataset = CustomDataset(data_tensor)

    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    return data_loader


if __name__ == "__main__":
    # 获取第一个批次
    dataloader = load_data(config.train_unknown_pt)
    print(len(dataloader))
    first_batch = next(iter(dataloader))

    print("Features in the first batch:")
    print(first_batch['features'])  # 打印特征
    print("Labels in the first batch:")
    print(first_batch['label'])  # 打印标签
