import colorama
import torch
import os
import time
from torch import nn, optim
import matplotlib.pyplot as plt
from datetime import datetime

from model.IO_NewUser import BinaryClassifier
from tools.load_data import load_data
from tools.config_file import NewUserPredictParams
from tools.gpu_setting import set_gpu
from tools.train import train
from utils_webhook.WeCom import WeCom


def create_figure_and_send_to_wecom(photo_save_path: str, data_list: list, webhook: WeCom):
    # 创建 x 轴数据（代表迭代次数或轮数）
    iterations = range(1, len(data_list) + 1)
    # 绘制折线图
    plt.plot(iterations, data_list, marker='o')
    # 添加标题和标签
    plt.title("Loss Trend Over Iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    # 显示网格线
    plt.grid()
    plt.savefig(photo_save_path)
    # 显示图形
    # plt.show()
    webhook.generate_img(plt_save_path)
    webhook.send()


def create_md_and_send_to_wecom(content: str, webhook: WeCom):
    webhook.generate_md(content=content)
    webhook.send()


if __name__ == "__main__":
    device = set_gpu()
    params = NewUserPredictParams()
    weCom = WeCom(params.we_com_webhook_url)
    # Updating below 2 params could change training
    data_num = 0
    model_name = "key2_key3_"

    # 将数据转换为合适的形状，即 (batch_size, input_size)
    data_loader = load_data(params.train_classified_pt[data_num], is_train=True)
    model = BinaryClassifier().to(device)
    print(model)
    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    start_time = time.time()
    loss_trend, best_loss = train(model=model, optimizer=optimizer, criterion=criterion, data_loader=data_loader,
                                  num_epochs=params.num_epochs, device=device)
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
    - Used time: {exhausted_time}\n
    - Best loss: {best_loss}\n
    """
    create_md_and_send_to_wecom(md_content, weCom)
