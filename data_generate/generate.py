import math

import pandas as pd
import torch
import colorama
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

from tools.config_file import NewUserPredictParams
from dataset import processing_time_stamp, normalize

params = NewUserPredictParams()


def generate_unknown():
    # Processing timestamp
    data = processing_time_stamp(params.train_unknown_csv)
    # Normalizing
    data_tensor = normalize(data, is_known=False)
    # Saving tensor to .pt
    torch.save(data_tensor, params.train_unknown_pt)

    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", params.train_unknown_pt)
    print(colorama.Fore.RESET)


def generate_known():
    pass


def generate_all():
    train_csv = processing_time_stamp(params.train_csv)
    train_csv_normalized = normalize(train_csv, False)
    torch.save(train_csv_normalized, params.train_all_pt)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", params.train_all_pt)
    print(colorama.Fore.RESET)


def generate_verify_test():
    test_csv = processing_time_stamp(params.test_csv)
    print(test_csv)
    test_csv_normalized = normalize(test_csv, False)
    print(test_csv_normalized)
    torch.save(test_csv_normalized, params.test_all_pt)
    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", params.test_all_pt)
    print(colorama.Fore.RESET)


# @TODO: To make another function to process dataset with known udmap
if __name__ == '__main__':
    generate_verify_test()
