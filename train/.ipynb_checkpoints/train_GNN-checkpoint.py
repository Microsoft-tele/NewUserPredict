import os
import sys

current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)

import pandas as pd
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import TUDataset
from networkx import from_pandas_edgelist

from tools.config_file import NewUserPredictParams

params = NewUserPredictParams()

if __name__ == "__main__":
    df_train = pd.read_csv(params.train_processed_csv)
    print(df_train)
    print(df_train)