# coding: utf-8

__all__ = ['EncoderRNN', 'BiAttention']


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy
import random


class EncoderRNN(nn.Module):
    def __init__(self, model_type, num_units, num_units_out, nlayers, bidir, dropout):
        super().__init__()
        if model_type == 'lstm':
            self.rnn = nn.LSTM(num_units, num_units_out//2 if bidir else num_units_out,
                               nlayers, batch_first=True, bidirectional=bidir,
                               dropout=dropout)
        elif model_type == 'gru':
            self.rnn = nn.GRU(num_units, num_units_out//2 if bidir else num_units_out,
                              nlayers, batch_first=True, bidirectional=bidir,
                              dropout=dropout)
        elif model_type == 'affine':
            self.linear = nn.Linear(num_units, num_units)
        else:
            raise NotImplementedError

    def forward(self, input, input_len=None, return_last=False):
        if getattr(self, 'rnn', None) is None:   # affine
            return self.linear(input)
        if input_len is None:
            output, _ = self.rnn(input)
            return output
        packed_input = pack_padded_sequence(input, input_len, batch_first=True,
                                            enforce_sorted=False)
        packed_output, (hidden, _) = self.rnn(packed_input)
        if return_last:
            return hidden.permute(1, 0, 2).contiguous().view(input.size(0), -1)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return output


class BiAttention(nn.Module):
    def __init__(self, input_size, dropout=0.):
        super().__init__()
        self.input_linear = nn.Linear(input_size, 1, bias=False)
        self.memory_linear = nn.Linear(input_size, 1, bias=False)
        self.dropout = dropout

        self.dot_scale = nn.Parameter(torch.Tensor(input_size).uniform_(1.0 / (input_size ** 0.5)))

    def forward(self, input, memory, mask):
        bsz, input_len, memory_len = input.size(0), input.size(1), memory.size(1)

        input = F.dropout(input, self.dropout)
        memory = F.dropout(memory, self.dropout)

        input_dot = self.input_linear(input)
        memory_dot = self.memory_linear(memory).view(bsz, 1, memory_len)
        cross_dot = torch.bmm(input * self.dot_scale, memory.permute(0, 2, 1).contiguous())
        att = input_dot + memory_dot + cross_dot   # B x N x N
        att = att.masked_fill(mask[:,:,None].repeat(1,1,memory_len), - 1e20)
        att = att.masked_fill(mask[:,None,:].repeat(1,input_len,1), - 1e20)

        weight_one = F.softmax(att, dim=-1)
        output_one = torch.bmm(weight_one, memory)
        weight_two = F.softmax(att.max(dim=-1)[0], dim=-1).view(bsz, 1, input_len)
        output_two = torch.bmm(weight_two, input)

        return torch.cat([input, output_one, input*output_one, output_two*output_one], dim=-1)
