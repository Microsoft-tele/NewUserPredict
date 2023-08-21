import torch
import pandas as pd

if torch.cuda.is_available():
    print("检测到当前设备有可用GPU:")
    print("当前可用GPU数量:", torch.cuda.device_count())
    print("当前GPU索引：", torch.cuda.current_device())
    print("当前GPU名称：", torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("未检测到当前设备有可用GPU，不建议开始训练，如有需求请自行更改代码：")
    exit()
