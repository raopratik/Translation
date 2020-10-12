import sys
sys.path.append(".")


import torch
import numpy as np
from v2 import con
import math
import torch.nn as nn
from transformers import XLMRobertaModel

# MODEL CONSTANTS
EMBEDDINGS_SIZE = 100
VOCAB_SIZE = 100
HIDDEN_SIZE = 100
LANG_VECTORS_SIZE = 50


class Encoder(torch.nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(Encoder, self).__init__()

        # Complete Transformer Encoder
        self.transformer_encoder = XLMRobertaModel.from_pretrained('xlm-roberta-base', return_dict=True)

        self.embedding = nn.Embedding(ntoken, ninp)

        self.ninp = ninp

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        output = self.transformer_encoder(src).last_hidden_state
        return output



