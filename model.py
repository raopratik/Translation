import sys

sys.path.append(".")

# import augment
import os
from collections import defaultdict
import time
import random
import torch
import numpy as np
import os
import numpy as np
import torch.nn.functional as F
from utilities.GradReverse import grad_reverse
from v2 import con
from torch.nn import TransformerEncoder, TransformerEncoderLayer, \
    TransformerDecoder, TransformerDecoderLayer
import math
import torch.nn as nn
from utilities.LangAdversarial import LangAdversarial
from utilities.SparseMax import SparsemaxLoss
from utilities.SparseMax_v2 import Sparsemaxv2
from utilities.LanguageEmbeddings import LanguageVectors

# MODEL CONSTANTS
EMBEDDINGS_SIZE = 100
VOCAB_SIZE = 100
HIDDEN_SIZE = 100
LANG_VECTORS_SIZE = 50


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        if con.CUDA:
            pe = pe.cuda()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        args = torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)

        div_term = np.exp(args.cpu().detach().numpy())

        pe[:, 0::2] = torch.tensor(np.sin(position.cpu().detach().numpy() * div_term))

        pe[:, 1::2] = torch.tensor(np.cos(position.cpu().detach().numpy() * div_term))

        pe = pe.unsqueeze(0).transpose(0, 1)

        if con.CUDA:
            pe = pe.cuda()

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        # x = self.dropout(x)

        return x


class Decoder(torch.nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(Decoder, self).__init__()

        self.pos_encoder = PositionalEncoding(ninp, dropout)

        # Encoder Layers
        decoder_layers = TransformerDecoderLayer(ninp, nhead, nhid, dropout)

        # Complete Transformer Encoder
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)

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

    def forward(self, tgt, memory, memory_key_padding_mask, tgt_mask,
                tgt_key_padding_mask=None):
        tgt = self.embedding(tgt.t()) * math.sqrt(self.ninp)
        tgt = self.pos_encoder(tgt)
        memory = memory.permute(1, 0, 2)

        if tgt_key_padding_mask is not None:
            output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask,
                                              tgt_key_padding_mask=tgt_key_padding_mask,
                                              memory_key_padding_mask=memory_key_padding_mask)
        else:
            output = self.transformer_decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask,
                                              memory_key_padding_mask=memory_key_padding_mask)

        return output


class Encoder(torch.nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(Encoder, self).__init__()

        self.pos_encoder = PositionalEncoding(ninp, dropout)

        # Encoder Layers
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)

        # Complete Transformer Encoder
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

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

    def forward(self, src, src_mask):
        src = self.embedding(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)

        return output


class ModelTETD(torch.nn.Module):
    def __init__(self, vocab_size, eos_id, predict_lang=False,
                 num_languages=2, lang_vectors=False,
                 sparse_flag=False, **kwargs):
        super(ModelTETD, self).__init__()
        global VOCAB_SIZE

        print("KWARGS", kwargs)

        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.lang_class_w = torch.nn.Parameter(torch.nn.init.xavier_uniform_(torch.empty(100 * 2, 4)),
                                               requires_grad=True)
        self.lang_vectors_flag = lang_vectors
        ntokens = vocab_size  # the size of vocabulary

        nlayers = kwargs.get('nlayers') or 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        nhid = kwargs.get('nhid') or 100  # the dimension of the feedforward network model in nn.TransformerEncoder
        emsize = kwargs.get('emsize') or 100  # embedding dimension
        nhead = kwargs.get('nhead') or 2  # the number of heads in the multiheadattention models
        dropout = kwargs.get('dropout') or 0.2  # the dropout value

        self.input_encoder = Encoder(ntokens, emsize, nhead, nhid, nlayers, dropout)
        self.tag_encoder = Encoder(ntokens, emsize, nhead, nhid, nlayers, dropout)

        self.decoder = Decoder(ntokens, emsize, nhead, nhid, nlayers, dropout)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

        self.predict_lang = predict_lang
        if predict_lang:
            self.lang_adversarial = LangAdversarial(state_size=HIDDEN_SIZE*2,
                                                    num_languages=num_languages)
        if sparse_flag:
            print("In Sparsemax Loss")
            self.criterion = SparsemaxLoss()

        self.fc = nn.Linear(emsize, vocab_size)

        if self.lang_vectors_flag:
            self.fc = torch.nn.Linear(emsize + LANG_VECTORS_SIZE,
                                          vocab_size)
            self.lang_vectors = LanguageVectors()




    def create_mask(self, src, src_lens):
        mask = []
        for i in range(len(src_lens)):
            mask.append([1 for _ in range(src_lens[i])] +
                        [0 for _ in range(src.shape[1] - src_lens[i])])
        mask = torch.tensor(mask).t()
        if con.CUDA:
            mask = mask.cuda()
        return mask

    def forward(self, input_dict, teacher_prob):
        """
        This function is just a place holder. I am returning just the loss and results"
        :param input_dict:
        :return:
        """

        input_ids = input_dict['input_ids']
        input_masks = input_dict['input_masks']
        tag_ids = input_dict['tag_ids']
        tag_masks = input_dict['tag_masks']
        output_ids = input_dict['output_ids']
        output_masks = input_dict['output_masks']
        output_seq_mask = input_dict['output_seq_masks']
        lang_ids = input_dict['lang_ids']
        lang_map = input_dict['language_map']

        # Run Encoder
        encoded_inputs = self.input_encoder(input_ids.t(), input_masks).permute(1, 0, 2)
        encoded_tags = self.tag_encoder(tag_ids.t(), tag_masks).permute(1, 0, 2)

        # print("encoded_inputs", encoded_inputs.shape)
        # print("encoded_tags", encoded_tags.shape)

        temp = torch.zeros((input_ids.shape[0], 1)).long()
        if con.CUDA:
            temp = temp.cuda()
        tgt = torch.cat((temp, input_dict['output_ids']), dim=1)[:, :-1]


        # tgt = torch.cat((torch.zeros((input_ids.shape[0], 1)).long(), output_ids), dim=1)[:, :-1]
        memory = torch.cat((encoded_inputs, encoded_tags), dim=1)
        tgt_mask = output_seq_mask
        tgt_key_padding_mask = output_masks
        memory_key_padding_mask = torch.cat((input_masks, tag_masks), dim=1)

        output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        if not self.lang_vectors_flag:
            output = self.fc(output)
        else:
            typological_vector = self.lang_vectors([lang_map[lang_id.cpu().detach().item()] for lang_id
                                                    in lang_ids])
            output = torch.cat((output,
                                   typological_vector.repeat(output.shape[0], 1, 1)),
                                   dim=2)
            output = self.fc(output)

        preds = torch.argmax(output, dim=2)
        loss = self.get_loss(input_dict, output)


        if self.predict_lang:
            loss += self.lang_adversarial(vectors=torch.cat((encoded_inputs.permute(1, 0, 2),
                                                             encoded_inputs.permute(1, 0, 2)),
                                                            dim=2),
                                          lang_id=lang_ids)

        return loss, preds

    def generate(self, input_dict):

        input_ids = input_dict['input_ids']
        input_masks = input_dict['input_masks']
        tag_ids = input_dict['tag_ids']
        tag_masks = input_dict['tag_masks']
        lang_ids = input_dict['lang_ids']
        lang_map = input_dict['language_map']

        # Run Encoder
        encoded_inputs = self.input_encoder(input_ids.t(), input_masks).permute(1, 0, 2)
        encoded_tags = self.tag_encoder(tag_ids.t(), tag_masks).permute(1, 0, 2)

        memory = torch.cat((encoded_inputs, encoded_tags), dim=1)
        memory_key_padding_mask = torch.cat((input_masks, tag_masks), dim=1)

        preds = torch.zeros(input_ids.shape).long()

        if con.CUDA:
            preds = preds.cuda()

        for i in range(input_ids.shape[1] - 1):
            tgt = preds[:, : i + 1]
            output_seq_mask = self.input_encoder._generate_square_subsequent_mask(i + 1)

            if con.CUDA:
                output_seq_mask = output_seq_mask.cuda()
            output = self.decoder(tgt=tgt, memory=memory, tgt_mask=output_seq_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

            if not self.lang_vectors_flag:
                output = self.fc(output)
            else:
                typological_vector = self.lang_vectors([lang_map[lang_id.cpu().detach().item()] for lang_id
                                                        in lang_ids])
                output = torch.cat((output,
                                    typological_vector.repeat(output.shape[0], 1, 1)),
                                   dim=2)
                output = self.fc(output)

            output = torch.argmax(output, dim=2)
            preds[:, 1:i + 2] = output.t()

        return preds[:, 1:].t()

    def generate_v2(self, input_dict):

        assert (not self.training)
        input_ids = input_dict['input_ids']
        input_masks = input_dict['input_masks']
        output_ids = torch.cat((torch.zeros((input_ids.shape[0], 1)).long(), input_dict['output_ids']), dim=1)[:, :-1]
        output_seq_mask = input_dict['output_seq_masks']

        src = self.embedding(input_ids.t()) * math.sqrt(self.emsize)

        src = self.pos_encoder(src)

        tgt = self.embedding(output_ids.t()) * math.sqrt(self.emsize)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(src, tgt, tgt_mask=output_seq_mask,
                                  src_key_padding_mask=input_masks,
                                  memory_key_padding_mask=input_masks)
        output = self.fc(output)
        output = torch.argmax(output, dim=2)
        return output

    def get_loss(self, input_dict, outputs):
        true_outputs = input_dict['output_ids'].reshape(-1)

        pred_outputs = outputs.permute(1, 0, 2).reshape(-1, self.vocab_size)

        if con.CUDA:
            true_outputs = true_outputs.cuda()
            pred_outputs = pred_outputs.cuda()
        loss = self.criterion(pred_outputs, true_outputs)
        return loss
