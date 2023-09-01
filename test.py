import os
import colorama
import torch
import matplotlib.pyplot as plt
import numpy as np
import time

from model.IO_BinaryClassifier import BinaryClassifierConfig
from tools import load_data, config_file
from tools.gpu_setting import set_gpu
from tools.test import test, select_model
from utils_webhook import WeCom

params = config_file.NewUserPredictParams()
weCom = WeCom.WeCom(params.we_com_webhook_url)

# set gpu id
device = set_gpu()


if __name__ == "__main__":
    model_name = select_model()
    model_path = os.path.join(params.model_save_path, model_name)
    model = torch.load(model_path).to(device)
    print(model)
    data_num = 0

    config_model = BinaryClassifierConfig()
    test_loader = load_data.load_data(pt_file_path=params.train_classified_pt[data_num],
                                      batch_size=config_model.batch_size, division_rate=config_model.division_rate,
                                      is_train=True)

    print("Loaded data:", params.train_classified_pt[data_num])

    precision, recall, f_score, accuracy = test(model=model, data_loader=test_loader, device=device)

    print(colorama.Fore.LIGHTGREEN_EX)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F_score:", f_score)
    print("Accuracy:", accuracy)
    content = f"""
    # Test result:
    - model name: {model_name}\n"
    - model structor:{str(model)}\n
    # Details:
    - Precision: {precision}\n
    - Recall: {recall}\n
    - F1_score: {f_score}\n
    - Accuracy: {accuracy}\n
    """

    weCom.generate_md(content)
    weCom.send()
    weCom.generate_img("./tmp.png")
    weCom.send()
