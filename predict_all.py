import os
import colorama
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd

from tools import load_data, config_file
from test_unknown import select_model

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

if __name__ == "__main__":
    model_path = os.path.join(paras.model_save_path, select_model())
    model = torch.load(model_path).to(device)
    print(model)
    pred_loader = load_data.load_all()

    # 创建一个空的列表，用于存储每次循环得到的预测结果
    y_pred_binary_list = []
    uuid_list = []
    for iterator in pred_loader:
        y_pred = model(iterator.to(device))
        y_pred = y_pred.squeeze()

        threshold = 0.5
        y_pred_binary = (y_pred >= threshold).float()

        y_pred_binary_np = y_pred_binary.detach().cpu().numpy()
        y_pred_binary_list.append(y_pred_binary_np.flatten())

    y_pred_binary_series = pd.Series(np.concatenate(y_pred_binary_list))
    # 创建一个 DataFrame，并将 Series 添加为一列
    result_df = pd.DataFrame({"all_predictions": y_pred_binary_series})
    # 添加递增的 uuid 列
    result_df.insert(0, "uuid", range(1, len(result_df) + 1))
    print(result_df)
    result_df.to_csv(paras.result_all_csv)
