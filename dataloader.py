import sys
sys.path.append(".")
from torch.utils import data
import numpy as np


class Translation(data.Dataset):
    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __len__(self):
        return len(self.input_dict["hrl"])

    def __getitem__(self, index):
        hrl = np.array(self.input_dict["hrl"][index])
        lrl = np.array(self.input_dict["lrl"][index])

        return hrl, lrl
