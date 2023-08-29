import os
import sys
import pandas as pd

import colorama
import torch

current_filename = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_filename)
great_parent_dir = os.path.dirname(parent_dir)
sys.path.append(great_parent_dir)

from tools.config_file import NewUserPredictParams

params = NewUserPredictParams()


