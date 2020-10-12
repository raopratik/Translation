import sys

sys.path.append(".")

import torch


class Translation(torch.nn.Module):
    def __init__(self, vocab_size=30522):
        super(Translation, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = torch.nn.Embedding(vocab_size, 100)
        self.rnn = torch.nn.LSTM(100, 100, batch_first=True)
        self.mlm = torch.nn.Linear(100, 30522)

    def forward(self, src, masked_lm_ids, masked_lm_positions, nsp_labels):
        embedded = self.embedding(src)
        outputs, _ = self.rnn(embedded)
        logits = self.mlm(outputs)
        return logits


