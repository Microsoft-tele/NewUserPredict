import time

import torch


def set_gpu() -> torch.device:
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
    print(type(device))
    print(device, "is available:")
    return device


if __name__ == '__main__':
    set_gpu()
