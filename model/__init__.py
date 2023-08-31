from torch import nn, optim

from . import *


class ConfigBase:
    def __init__(self):
        self.input_dim = 14
        self.output_dim = 1
        self.batch_size = 512
        self.lr = 0.001
        self.epoch_num = 1000
        self.division_rate = 0.8
