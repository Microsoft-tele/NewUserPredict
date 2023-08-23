import os
import colorama
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from tools import load_data, config_file

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


if __name__ == "__main__":
    model_path = os.path.join(paras.model_save_path, select_model())
    model = torch.load(model_path).to(device)
    print(model)
    test_loader = load_data.load_data(is_train=False)

    total_non_zero = 0
    total_test = 0
    for iterator in test_loader:
        y_pred = model(iterator["features"].to(device))
        y_pred = y_pred.squeeze()

        threshold = 0.5
        y_pred_binary = (y_pred >= threshold).float()
        # print(y_pred)
        # print(iterator["label"])
        # print("shape of y_pred:", y_pred.shape)
        # print(iterator["label"].shape)
        cha = iterator["label"].to(device) - y_pred_binary
        # print(cha.shape)
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
        plt.show()
        break
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Accuracy of model:", 1 - (total_non_zero / total_test))