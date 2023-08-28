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
from utils_webhook import WeCom


def show_figure_and_send_to_wecom(plt_save_path: str, data_list: list):
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
    plt.savefig(plt_save_path)
    # 显示图形
    plt.show()


if __name__ == "__main__":
    device = set_gpu()
    params = NewUserPredictParams()

    # 将数据转换为合适的形状，即 (batch_size, input_size)
    data_loader = load_data(params.train_classified_pt[0], is_train=True)
    # 使用模型进行预测
    # 输入特征的维度
    # 创建模型实例
    model = BinaryClassifier().to(device)
    print(model)
    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    loss_trend = []

    start_time = time.time()

    md_content = """
    # Train log
    ## Loss of every epoch\n
    """

    for epoch in range(params.num_epochs):
        print(f"Epoch {epoch + 1}/{params.num_epochs}")
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
            # 打印损失
        batch_loss = sum(loss_epoch) / len(loss_epoch)
        print("Batch Loss:", batch_loss)
        loss_trend.append(batch_loss)

    end_time = time.time()
    exhausted_time = end_time - start_time
    print("Total time:", exhausted_time / 60)

    current_time = datetime.now()
    # 格式化时间为年月日时分
    formatted_time = current_time.strftime("%Y_%m_%d_%H_%M")

    model_name = "key_key3"
    plt_save_path = os.path.join(params.plt_save_path, model_name + formatted_time + ".png")

    show_figure_and_send_to_wecom(plt_save_path=plt_save_path, data_list=loss_trend)

    model_save_path = os.path.join(params.model_save_path, model_name + formatted_time + ".pkl")

    torch.save(model, model_save_path)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Training finish!!!")
    print("Model has been saved in:", model_save_path)
    print("Loss trend figure has been saved in:", plt_save_path)
    print(colorama.Fore.RESET)

    dt_start = datetime.fromtimestamp(start_time)
    formatted_start_time = dt_start.strftime("%Y/%m/%d %H:%M")

    dt_end = datetime.fromtimestamp(end_time)
    formatted_end_time = dt_end.strftime("%Y/%m/%d %H:%M")

    md_content += "## Start time: " + str(formatted_start_time) + "\n"
    md_content += "## End time: " + str(formatted_end_time) + "\n"
    md_content += "## Used time: " + str(exhausted_time / 60) + " min\n"
    md_content += "## Final Loss:" + str(batch_loss) + "\n"

    weCom = WeCom.WeCom("https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=a7f64d24-662b-46ba-bbaf-cdf3f6209727")
    weCom.generate_md(content=md_content)
    weCom.send()
    weCom.generate_img(plt_save_path)
    weCom.send()
