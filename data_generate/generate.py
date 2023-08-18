import pandas as pd
import torch
import colorama


# 读取 CSV 文件，不将第一行视为列名
def generate_unknown(filepath: str, save_path: str):
    data = pd.read_csv(filepath)

    # 删除第0列和第2列
    data = data.drop(data.columns[[0, 2]], axis=1)

    # 去掉第一行（列名）
    data = data.iloc[1:]

    # 重置索引，以便重新设置正确的行索引
    data = data.reset_index(drop=True)

    data_numpy = data.values

    data_tensor = torch.from_numpy(data_numpy).float()
    torch.save(data_tensor, save_path)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print(colorama.Fore.RESET)


if __name__ == '__main__':
    generate_unknown("../dataset/train_unknown.csv", "./train_unknown.pt")
