import os
import colorama
import torch
import matplotlib.pyplot as plt
import numpy as np

from tools import load_data, config_file

paras = config_file.NewUserPredictParams()


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
    model = torch.load(model_path)
    print(model)
    test_loader = load_data.load_data(is_train=False)

    for iterator in test_loader:
        y_pred = model(iterator["features"])
        y_pred = y_pred.squeeze()
        # print(y_pred)
        # print(iterator["label"])
        print(y_pred.shape)
        print(iterator["label"].shape)

        cha = iterator["label"] - y_pred
        print(cha.shape)

        cha_numpy = cha.detach().numpy()

        x = np.arange(len(cha_numpy))
        # 绘制折线图
        plt.plot(x, cha_numpy)

        # 添加标题和标签
        plt.title('Line chart')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')

        # 显示图形
        plt.show()
        break
