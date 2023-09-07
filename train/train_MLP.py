import os
import sys

import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader

current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)

import colorama
import torch
import os
import time
from torch import nn, optim
from datetime import datetime

from model.IO_BinaryClassifier import BinaryClassifier, BinaryClassifierConfig
from tools.config_file import NewUserPredictParams
from tools.gpu_setting import set_gpu
from tools.train import train
from utils_webhook.WeCom import WeCom
from utils_webhook.WeCom import create_figure_and_send_to_wecom, create_md_and_send_to_wecom


class MLPDataset(Dataset):
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


def load_data_mlp(tensor_dataset: torch.Tensor, division_rate, batch_size: int, is_train=True):
    dataset_train = MLPDataset(tensor_dataset)
    # 计算分割索引

    total_samples = len(dataset_train)
    train_size = int(division_rate * total_samples)

    train_subset = Subset(dataset_train, range(train_size))
    test_subset = Subset(dataset_train, range(train_size, total_samples))

    # 创建数据加载器
    if is_train:
        return DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    else:
        return DataLoader(test_subset, batch_size=batch_size, shuffle=False)  # 不需要在测试时打乱顺序


if __name__ == "__main__":
    model_name = "MLP_"
    device = set_gpu()
    params = NewUserPredictParams()
    weCom = WeCom(params.we_com_webhook_url)
    df_train = pd.read_csv(params.train_processed_csv)
    pd.set_option("display.max_columns", None)
    print("All columns:")
    print(df_train.columns)
    features = [
        'uuid', 'eid', 'eid_target', 'common_ts', 'date', 'hour', 'weekday', 'sin_norm', 'cos_norm', 'x2', 'x4', 'x5',
        'x6', 'x7', 'x8', 'target']
    df_train_selected = df_train[features]
    print("Selected columns")
    print(df_train_selected.columns)
    tensor_train_selected = torch.tensor(df_train_selected.to_numpy(), dtype=torch.float32)

    model_config = BinaryClassifierConfig()

    dataloader = load_data_mlp(tensor_train_selected, 0.8, model_config.batch_size, True)

    model = BinaryClassifier(model_config).to(device)

    print(model)
    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失函数
    # @TODO: add lr to config file
    optimizer = optim.Adam(model.parameters(), lr=model_config.lr)

    start_time = time.time()
    loss_trend, best_loss = train(model=model, optimizer=optimizer, criterion=criterion, data_loader=dataloader,
                                  num_epochs=model_config.epoch_num, device=device)
    end_time = time.time()
    exhausted_time = end_time - start_time
    print("Total time:", exhausted_time / 60)

    start_date_time = datetime.fromtimestamp(start_time)
    formatted_start_time = start_date_time.strftime("%Y-%m-%d--%H-%M_")
    end_date_time = datetime.fromtimestamp(end_time)
    formatted_end_time = end_date_time.strftime('%Y-%m-%d--%H-%M_')

    plt_save_path = os.path.join(params.plt_save_path, formatted_end_time + model_name + ".png")
    model_save_path = os.path.join(params.model_save_path, formatted_end_time + model_name + ".pkl")
    create_figure_and_send_to_wecom(photo_save_path=plt_save_path, data_list=loss_trend, webhook=weCom)

    torch.save(model, model_save_path)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Training finish!!!")
    print("Model has been saved in:", model_save_path)
    print("Loss trend figure has been saved in:", plt_save_path)
    print(colorama.Fore.RESET)

    md_content = f"""
    # Train log:\n
    - model name: {model_name}\n
    # Details:\n
    - Start time: {formatted_start_time}\n
    - End time: {formatted_end_time}\n
    - Used time: {exhausted_time / 60}\n
    - Best loss: {best_loss}\n
    """
    create_md_and_send_to_wecom(md_content, weCom)
