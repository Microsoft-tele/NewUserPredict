import math

import pandas as pd
import torch
import colorama
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F

from config import Config
from dataset import processing_time_stamp, normalize

config = Config()


def generate_unknown():
    # Processing timestamp
    data = processing_time_stamp()
    # Normalizing
    data_tensor = normalize(data, is_known=False)
    # Saving tensor to .pt
    torch.save(data_tensor, config.train_unknown_pt)

    print(colorama.Fore.LIGHTGREEN_EX)
    print("Convert dataset successfully!!!")
    print("You can search your .pt at:", config.train_unknown_pt)
    print(colorama.Fore.RESET)


# @TODO: To make another function to process dataset with known udmap
if __name__ == '__main__':
    generate_unknown()
