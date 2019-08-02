# coding = utf-8

# @time    : 2019/7/8 2:46 PM
# @author  : alchemistlee
# @fileName: rank_net.py
# @abstract:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy
import random
from pytorch_pretrained_bert import BertModel, BertForMaskedLM
from siamese.nn_utils import EncoderRNN
import pdb


class SiameseModel(nn.Module):

    def forward_pair(self, x):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError

    def compute_loss(self, x):
        raise NotImplementedError

    def inference(self, x):
        raise NotImplementedError


class RankNet(SiameseModel):

    __constants__ = ['nvocab', 'dim', 'dropout', 'loss_type', 'margin',
                     'bilinear', 'enable_bert']

    def __init__(self, nvocab, dim, dropout=0.0, loss_type='siamese',
                 margin=1., enable_bert=False, max_sent_len=512,
                 max_sent_count=100):
        """
        Constrcutor for RankNet

        Args:
            nvocab:
            dim:
            dropout:
            loss_type:
            margin:
            bilinear:
            enable_bert:
        """
        super().__init__()

        self.enable_bert = enable_bert
        if self.enable_bert:
            self.bert_model = BertModel.from_pretrained('bert-base-chinese').to('cuda')
            self.bert_model.eval()
            dim = 768
        self.pad_sent_cnt = max_sent_count
        self.pad_sent_len = max_sent_len

        self.emb = nn.Embedding(nvocab, dim)
        self.sent_pos_emb = nn.Embedding(self.pad_sent_cnt+1, dim)
        self.rnn = EncoderRNN('lstm', dim, dim // 2, 2, True, dropout)
        self.loss_type = loss_type
        self.margin = margin

        self.fout = nn.Linear(dim, 1)

        print('init model complete ... ')

    def forward(self, x, x_len, sent_idx):
        """
        Forward function on normal articles

        Args:
            x          : B x N x L
            x_len      : B x N
            sent_idx   : B x N
        Returns:
            output     : B x N
        """
        B = x.size(0)
        N = x.size(1)
        L = x.size(2)

        # produce sent_mask
        sent_mask = x_len == self.pad_sent_len
        x_len = x_len.masked_fill(sent_mask, 1)

        # emb
        if self.enable_bert:
            with torch.no_grad():
                _x = torch.Tensor(B, N, L, 768).zero_().to('cuda')
                for b in range(B):
                    this_n_sent = sent_mask[b].eq(1).sum().item()
                    for n in range(this_n_sent):
                        this_x = x[b, n][:x_len[b, n]]
                        this_x_bert, _ = self.bert_model(this_x.unsqueeze(0).to('cuda'))
                        this_x_bert = this_x_bert[-1].detach()
                        _x[b, n][:x_len[b, n]].copy_(this_x_bert.squeeze(0))
                x = _x  # B x N x L x D
        else:
            x = self.emb(x)  # B x N x L x D
        x = x.view(B * N, L, -1)
        x = self.rnn(x, input_len=x_len.view(-1), return_last=True)  # (B * N) x D   # TODO do we need an extra mask?

        # processing indices
        x_pos = self.sent_pos_emb(sent_idx).view(B * N, -1)  # (B * N) x D
        x = x + x_pos

        x = self.fout(x).view(B, N)  # B x N
        x = x.masked_fill(sent_mask, - 1e20)
        return x

    def forward_pair(self, x, x_len, sent_idx):
        """
        Forward function on a pair

        Args:
            x          : [[B x L], [B x L]]
            x_len      : [[B], [B]]
            sent_idx : [[B] , [B]]

        Returns:
            output     : B x 2
        """
        x_pos = x[0]  # B x L
        x_neg = x[1]  # B x L

        x_len_pos = x_len[0]  # B
        x_len_neg = x_len[1]  # B

        sent_idx_pos = sent_idx[0]  # B
        sent_idx_neg = sent_idx[1]  # B

        B, L = x_pos.size(0), x_pos.size(1)

        # emb
        if self.enable_bert:
            with torch.no_grad():
                _x_pos = torch.Tensor(B, L, 768).zero_().to('cuda')
                for b in range(B):
                    this_x_pos = x_pos[b][:x_len_pos[b]]
                    this_x_pos_bert, _ = self.bert_model(this_x_pos.unsqueeze(0).to('cuda'))
                    this_x_pos_bert = this_x_pos_bert[-1].detach()
                    _x_pos[b][:x_len[b]].copy_(this_x_pos_bert.squeeze(0))
                x_pos = _x_pos

                _x_neg = torch.Tensor(B, L, 768).zero_().to('cuda')
                for b in range(B):
                    this_x_neg = x_neg[b][:x_len_neg[b]]
                    this_x_neg_bert, _ = self.bert_model(this_x_neg.unsqueeze(0).to('cuda'))
                    this_x_neg_bert = this_x_neg_bert[-1].detach()
                    _x_neg[b][:x_len[b]].copy_(this_x_neg_bert.squeeze(0))
                x_neg = _x_neg

        else:
            x_pos = self.emb(x_pos)  # B x L x D
            x_neg = self.emb(x_neg)  # B x L x D

        # concat tenor
        x_all = torch.cat((x_pos, x_neg), 0)  # 2*B x L x D
        x_len_all = torch.cat((x_len_pos, x_len_neg), 0)  # 2*B

        # position emb
        sent_idx_pos = self.sent_pos_emb(sent_idx_pos)  # B x D
        sent_idx_neg = self.sent_pos_emb(sent_idx_neg)  # B x D

        # position embedding
        sent_idx_all = torch.cat((sent_idx_pos, sent_idx_neg), 0)  # 2*B x D

        # representation
        x_all = self.rnn(x_all, input_len=x_len_all, return_last=True)  # 2*B x D
        x_all = self.fout(x_all + sent_idx_all)  # 2*B

        res = torch.split(x_all, B, dim=0)
        res_pos = res[0]
        res_neg = res[1]
        return torch.cat((res_pos, res_neg), dim=1)

    def compute_loss_pair(self, x, x_len, sent_idx):
        """
        Compute loss on the pair of data

        Args:
            x          : [[B x L], [B x L]]
            x_len      : [[B], [B]]
            sent_idx : [[B] , [B]]
        Returns:
            loss       : [1]
        """
        B = x[0].size(0)
        out = self.forward_pair(x, x_len, sent_idx)
        if self.loss_type == 'softmax':
            y_label = torch.LongTensor(B).to('cuda').fill_(0)
            return F.cross_entropy(out, y_label)
        elif self.loss_type == 'siamese':
            y_label = torch.Tensor(B).to('cuda').fill_(1)
            return F.margin_ranking_loss(out[:, 0], out[:, 1],
                                         y_label, self.margin)
        else:
            raise NotImplementedError

    def inference(self, x, x_len, sent_idx, top_n=5):
        """
        Inference loop, taking the top_n

        Args:
            x          : B x N x L
            x_len      : B x N
            sent_idx   : B x N
            top_n      : int

        Returns:
            res        : B x top_n
        """

        out = self.forward(x, x_len, sent_idx)  # B x N
        sent_mask = x_len == self.pad_sent_len
        res = []
        for b in range(len(out)):
            item_len = sent_mask[b].eq(0).sum().item()
            item = out[b][:item_len]
            item_ids = item.detach().cpu().numpy().tolist()

            item_ids = numpy.argsort(item_ids)[::-1]
            item_ids = item_ids[:top_n]
            item_topn_idx = sent_idx[b].cpu().numpy()[item_ids]
            res.append(item_topn_idx)
        return res

