import os
import sys

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

from model.IO_BinaryClassifierEmbedding import BinaryClassifierEmbedding, BinaryClassifierEmbeddingConfig, \
    load_data_from_csv
from tools.config_file import NewUserPredictParams
from tools.gpu_setting import set_gpu
from tools.train import train
from tools.test import select_model, test
from utils_webhook.WeCom import WeCom
from utils_webhook.WeCom import create_figure_and_send_to_wecom, create_md_and_send_to_wecom

if __name__ == "__main__":
    device = set_gpu()
    params = NewUserPredictParams()
    model_config = BinaryClassifierEmbeddingConfig()
    weCom = WeCom(params.we_com_webhook_url)
    # Updating below 2 params could change training
    model_name = "embedding_"

    # 将数据转换为合适的形状，即 (batch_size, input_size)
    data_loader = load_data_from_csv(train_dataset_path=params.train_processed_csv,
                                     test_dataset_path=params.test_processed_csv, batch_size=model_config.batch_size,
                                     is_train=1)
    model = torch.load(os.path.join(params.model_save_path, select_model()))
    print(model)
    # Log start time
    start_time = time.time()
    precision, recall, f_score, accuracy = test(model=model, data_loader=data_loader, device=device)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F_score:", f_score)
    print("Accuracy:", accuracy)
    end_time = time.time()

