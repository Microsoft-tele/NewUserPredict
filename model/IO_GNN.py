import os
import sys

current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from tools.gpu_setting import set_gpu

if __name__ == "__main__":
    set_gpu()
