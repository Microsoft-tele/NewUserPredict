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
from utils_webhook.WeCom import WeCom
from utils_webhook.WeCom import create_figure_and_send_to_wecom, create_md_and_send_to_wecom

if __name__ == "__main__":
    best_loss = None
    loss_trend = None
    device = set_gpu()
    params = NewUserPredictParams()
    model_config = BinaryClassifierEmbeddingConfig()
    weCom = WeCom(params.we_com_webhook_url)
    # Updating below 2 params could change training

    model_name = "embedding_"

    # 将数据转换为合适的形状，即 (batch_size, input_size)
    data_loader = load_data_from_csv(train_dataset_path=params.train_processed_csv,
                                     test_dataset_path=params.test_processed_csv, batch_size=model_config.batch_size,
                                     is_train=0)
    model = BinaryClassifierEmbedding(model_config).to(device)
    print(model)
    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=model_config.lr)

    # Log start time
    start_time = time.time()
    try:
        loss_trend, best_loss = train(model=model, optimizer=optimizer, criterion=criterion, data_loader=data_loader,
                                      num_epochs=model_config.epoch_num, device=device)

        end_time = time.time()
        exhausted_time = end_time - start_time
        print("Total time:", exhausted_time / 60)
        # 格式化时间为年月日时分

        start_date_time = datetime.fromtimestamp(start_time)
        formatted_start_time = start_date_time.strftime("%Y-%m-%d--%H-%M_")
        end_date_time = datetime.fromtimestamp(end_time)
        formatted_end_time = end_date_time.strftime('%Y-%m-%d--%H-%M_')

        plt_save_path = os.path.join(params.plt_save_path, formatted_end_time + model_name + ".png")
        model_save_path = os.path.join(params.model_save_path, formatted_end_time + model_name + ".pkl")

        # Save model to pkl
        torch.save(model, model_save_path)
        print(colorama.Fore.LIGHTGREEN_EX)
        print("Training finish!!!")
        print("Model has been saved in:", model_save_path)
        print("Loss trend figure has been saved in:", plt_save_path)
        print(colorama.Fore.RESET)

        create_figure_and_send_to_wecom(photo_save_path=plt_save_path, data_list=loss_trend, webhook=weCom)

        md_content = f"""
        # Train log:\n
        - model name: {model_name}\n
        # Details:\n
        - Start time: {formatted_start_time}\n
        - End time: {formatted_end_time}\n
        - Used time: {exhausted_time}\n
        - Best loss: {best_loss}\n
        """
        create_md_and_send_to_wecom(md_content, weCom)

    except KeyboardInterrupt:
        print(colorama.Fore.LIGHTYELLOW_EX)
        print("Notice: You have interrupted training. Please confirm whether saving current model and send log to "
              "wecom:(Y/n)")
        print(colorama.Fore.RESET)
        op = input()
        if op == 'Y':
            end_time = time.time()
            exhausted_time = end_time - start_time
            print("Total time:", exhausted_time / 60)
            # 格式化时间为年月日时分

            start_date_time = datetime.fromtimestamp(start_time)
            formatted_start_time = start_date_time.strftime("%Y-%m-%d--%H-%M_")
            end_date_time = datetime.fromtimestamp(end_time)
            formatted_end_time = end_date_time.strftime('%Y-%m-%d--%H-%M_')

            plt_save_path = os.path.join(params.plt_save_path, formatted_end_time + model_name + ".png")
            model_save_path = os.path.join(params.model_save_path, formatted_end_time + model_name + ".pkl")

            # Save model to pkl
            torch.save(model, model_save_path)
            print(colorama.Fore.LIGHTGREEN_EX)
            print("Training interrupted!!!")
            print("Model has been saved in:", model_save_path)
            print("Loss trend figure has been saved in:", plt_save_path)
            print(colorama.Fore.RESET)

        else:
            print(colorama.Fore.LIGHTYELLOW_EX)
            print("Exit training process!!!")
            print(colorama.Fore.RESET)
    finally:
        print(colorama.Fore.LIGHTWHITE_EX)
        print("Training over!!!")
        print(colorama.Fore.RESET)

