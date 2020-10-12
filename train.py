import sys

sys.path.append(".")

import torch
from preprocess import Preprocess
from model import Encoder
import constants as con
from torch import optim


class PretrainingTrainer:
    def __init__(self):
        self.preprocessor = None
        self.model = None
        self.optimizer = None

    def setup_preprocessed_data(self):
        self.preprocessor = Preprocess()
        self.preprocessor.setup()

    def setup_model(self):
        # Create multilingual vocabulary
        self.model = Encoder()

        if con.CUDA:
            self.model = self.model.cuda()

    def setup_scheduler_optimizer(self):
        lr_rate = 0.001
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr_rate, weight_decay=0)

    def train_model(self):
        train_loader = self.preprocessor.train_loaders
        batch_size = 8

        self.model.train()
        train_loss = 0
        batch_correct = 0
        total_correct = 0
        index = 0
        for hrl_src, lrl_src, hrl_att, lrl_att in train_loader:
            logits = self.model(hrl_src)
            print(logits.shape)
            break
            # self.optimizer.zero_grad()
            # batch_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            # self.optimizer.step()
            # batch_correct += self.evaluate(masked_outputs=masked_outputs, masked_lm_ids=masked_lm_ids)
            # total_correct += (8 * 20)

    def run_pretraining(self):
        self.setup_preprocessed_data()
        self.setup_model()
        self.setup_scheduler_optimizer()
        self.train_model()


if __name__ == '__main__':
    trainer = PretrainingTrainer()
    trainer.run_pretraining()

