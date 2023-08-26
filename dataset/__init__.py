import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from tools import config_file, processing_time_stamp, normalize
from datetime import datetime

params = config_file.NewUserPredictParams()

if __name__ == "__main__":
    data = processing_time_stamp()
    data_tensor = normalize(data, is_known=False)
    torch.save(data_tensor, params.train_unknown_pt)
