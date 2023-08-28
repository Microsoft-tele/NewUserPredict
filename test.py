import os
import colorama
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from tools import load_data, config_file
from utils_webhook import WeCom

paras = config_file.NewUserPredictParams()

# set gpu id
if torch.cuda.is_available():
    print("检测到当前设备有可用GPU:")
    print("当前可用GPU数量:", torch.cuda.device_count())
    print("当前GPU索引：", torch.cuda.current_device())
    print("当前GPU名称：", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("未检测到当前设备有可用GPU，不建议开始训练，如有需求请自行更改代码：")
    exit()

# torch.cuda.set_device(args.device_id)
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
device = torch.device("cuda")  # 默认使用GPU进行训练
print(device, "is available:")


def select_model():
    file_list = os.listdir(paras.model_save_path)

    print(colorama.Fore.LIGHTGREEN_EX)

    for i in range(len(file_list)):
        print(f"{i}:{file_list[i]}")

    print("Please select a model:")
    print(colorama.Fore.RESET)

    op = -1
    while True:
        try:
            op = input()
            op = int(op)
            break
        except:
            print(colorama.Fore.LIGHTRED_EX)
            print("Please input again:")
            print(colorama.Fore.RESET)
            continue
    return file_list[op]


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

    return precision, recall, f_score


if __name__ == "__main__":
    model_name = select_model()
    model_path = os.path.join(paras.model_save_path, model_name)
    model = torch.load(model_path).to(device)
    print(model)
    test_loader = load_data.load_data(is_train=False)

    total_non_zero = 0
    total_test = 0
    for iterator in test_loader:
        # print(iterator)
        # print(iterator["features"])
        y_pred = model(iterator["features"].to(device))
        y_pred = y_pred.squeeze()

        threshold = 0.5
        y_pred_binary = (y_pred >= threshold).float()
        # print(y_pred)
        # print(iterator["label"])
        # print("shape of y_pred:", y_pred.shape)
        # print(iterator["label"].shape)
        cha = iterator["label"].to(device) - y_pred_binary
        precision, recall, f_score = F_score(iterator["label"].to(device), y_pred_binary)
        print(colorama.Fore.LIGHTGREEN_EX)
        print("precision: ", precision)
        print("recall: ", recall)
        print("f_score: ", f_score)
        print(colorama.Fore.RESET)

        cha_numpy = cha.detach().cpu().numpy()
        non_zero_count = np.count_nonzero(cha_numpy)
        total_non_zero += non_zero_count
        total_test += len(cha_numpy)
        # print("Number of non-zero elements:", non_zero_count)
        # print("Accuracy of model:", 1 - (non_zero_count / len(cha_numpy)))

        x = np.arange(len(cha_numpy))
        # 绘制折线图
        plt.scatter(x, cha_numpy)
        # 添加标题和标签
        plt.title('Line chart')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')

        # 显示图形
        plt.savefig("./tmp.png")
        plt.show()
        break
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Accuracy of model:", 1 - (total_non_zero / total_test))

    weCom = WeCom.WeCom(paras.we_com_webhook_url)
    content = "# Test result:\n"
    content += str(model) + "\n"
    content += f"# Accuracy of {model_name}: {1 - (total_non_zero / total_test)}"
    weCom.generate_md(content)
    weCom.send()
    weCom.generate_img("./tmp.png")
    weCom.send()
