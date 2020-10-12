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
        hrl_src = np.array(self.input_dict["hrl"]['input_ids'][index])
        lrl_src = np.array(self.input_dict["lrl"]['input_ids'][index])
        hrl_att = np.array(self.input_dict["hrl"]['attention_ids'][index])
        lrl_att = np.array(self.input_dict["lrl"]['attention_ids'][index])

        return hrl_src, lrl_src, hrl_att, lrl_att
