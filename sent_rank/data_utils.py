# coding: utf-8

import pickle as pkl
import torch
import os, random, math
import collections
import numpy
import jieba


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def get_words_idx(self, word_toks):
        res = []
        for tk in word_toks:
            res.append(self.__call__(tk))
        return res


def build_vocab(content, max_size=50000, min_freq=0):
    """return a dictionary of max_size with words of freq higher than min_freq"""
    counter = collections.Counter()
    for i, seq in enumerate(content):
        # counter.update(jieba.cut(seq))
        seq = seq.split()
        counter.update(seq)
        if i % 1000 == 0:
            print(f"Buidling Vocab: {i} has been done.")

    words = []
    for sym, cnt in counter.most_common(max_size):
        if cnt < min_freq:
            break
        words.append(sym)

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


class DataLoader(object):
    def __init__(self, train_dir=None, test_dir=None, val_dir=None,
                 all_dir=None, max_sent_len=512, max_sent_count=100,
                 vocab_path='./vocab.pth', enable_bert=False,
                 filter_valid=True, logger=None):
        """
        Constructor of DataLoader

        Args:
            train_dir:
            test_dir:
            val_dir:
            all_dir:
            max_sent_len:
            max_sent_count:
            vocab_path:
        """

        self.enable_bert = enable_bert
        self.filter_valid = filter_valid  # TODO
        self.max_sent_count = max_sent_count
        self.max_sent_len = max_sent_len
        self.logger = logger

        self.logger.info('train mode ...')
        if all_dir is None:
            with open(train_dir, 'rb') as ft:
                ori_train = pkl.load(ft)
            with open(test_dir, 'rb') as ft:
                ori_test = pkl.load(ft)
            with open(val_dir, 'rb') as ft:
                ori_val = pkl.load(ft)
        else:
            with open(all_dir, 'rb') as ft:
                ori_all = pkl.load(ft)
            ori_train, ori_val, ori_test = self._split_train_val_test(ori_all)

        # filter  TODO debug mode
        '''
        ori_train = ori_train[:100]
        ori_test = ori_test[:100]
        ori_val = ori_val[:100]
        '''
        if self.filter_valid:   # TODO
            ori_train = self._filter(ori_train)
            ori_test = self._filter(ori_test)
            ori_val = self._filter(ori_val)

        # embeding
        if not self.enable_bert:
            def _ext_all_sent(input):
                res = []
                for item in input:
                    res.extend(item['sent'])
                return res

            all_content = []
            all_content.extend(_ext_all_sent(ori_train))
            all_content.extend(_ext_all_sent(ori_test))
            all_content.extend(_ext_all_sent(ori_val))

            if not vocab_path is None and os.path.exists(vocab_path):
                self.vocab = torch.load(vocab_path)
            else:
                self.vocab = build_vocab(all_content, min_freq=3)
                torch.save(self.vocab, 'vocab.pth')
            self.logger.info('built vocabulary size = {} !'.format(len(self.vocab)))

            self.emb_train = self._add_emb(ori_train)
            self.emb_test = self._add_emb(ori_test)
            self.emb_val = self._add_emb(ori_val)
        else:
            self.emb_train = self._add_emb_bert(ori_train)
            self.emb_test = self._add_emb_bert(ori_test)
            self.emb_val = self._add_emb_bert(ori_val)

        self.data_iter = {'train': 0, 'test': 0, 'val': 0}
        self.data_len = {'train': len(self.emb_train),
                         'test' : len(self.emb_test),
                         'val'  : len(self.emb_val)}

        self.logger.info('init finished ... ')

    def get_iter_batch(self, data_type='train', batch_size=64):
        """
        Get an ordered iterated batch

        Args:
            set: ['train'|'val'|'test']
            batch_size:

        Returns:
            x           : [B x N x L]
            sent_in_abs : [B x N]
            sent_len    : [B x N]
            x_ori_sent  : list of str
        """
        if data_type == 'train':
            data_set = self.emb_train
        elif data_type == 'val':
            data_set = self.emb_val
        elif data_type == 'test':
            data_set = self.emb_test
        elif isinstance(data_type, list):   # TODO ugly
            ''' self-constructed batch '''
            data_set = data_type

        x, sent_idx, sent_in_abs, sent_len, x_ori_sent = [], [], [], [], []
        batch_max_sent_count, batch_max_sent_len = 0, 0

        for _ in range(batch_size):
            if isinstance(data_type, list):   # TODO ugly
                item = data_set[0]
                assert(batch_size == 1)
            else:
                try:
                    item = data_set[self.data_iter[data_type]]
                    self.data_iter[data_type] += 1
                except IndexError:
                    self.data_iter[data_type] = 0
                    return None

            S = len(item['sent'])
            assert S == len(item['sent_idx'])
            assert S == len(item['sent_in_abs'])
            assert S == len(item['sent_emb'])

            if S >= self.max_sent_count:
                continue

            if batch_max_sent_count < S:
                batch_max_sent_count = S

            _x = []
            _sent_idx = []
            _sent_in_abs = []
            _sent_len = []
            _x_ori_sent = {}
            for i in range(S):
                item_sent = (item['sent'])[i]
                item_emb = (item['sent_emb'])[i]
                item_sent_idx = (item['sent_idx'])[i]
                item_abs_idx = (item['sent_in_abs'])[i]

                if len(item_emb) > batch_max_sent_len:
                    batch_max_sent_len = len(item_emb)
                elif len(item_emb) == 0:
                    raise ValueError

                _x.append(item_emb)
                _sent_idx.append(item_sent_idx)
                _sent_in_abs.append(item_abs_idx)
                _sent_len.append(len(item_emb))
                _x_ori_sent[item_sent_idx] = item_sent

            x.append(_x)
            sent_idx.append(_sent_idx)
            sent_in_abs.append(_sent_in_abs)
            sent_len.append(_sent_len)
            x_ori_sent.append(_x_ori_sent)

        # built tensor with padding
        B = len(x)
        t_x = torch.Tensor(B, batch_max_sent_count, batch_max_sent_len).zero_()
        t_sent_idx = torch.Tensor(B, batch_max_sent_count)\
                          .fill_(self.max_sent_count)
        t_sent_len = torch.Tensor(B, batch_max_sent_count)\
                          .fill_(self.max_sent_len)
        t_sent_in_abs = torch.Tensor(B, batch_max_sent_count).fill_(-1)
        for b in range(B):
            for ii, s in enumerate(x[b]):
                t_x[b, ii, :len(s)].copy_(torch.tensor(s))
            t_sent_idx[b, :len(sent_idx[b])].copy_(torch.tensor(sent_idx[b]))
            t_sent_len[b, :len(sent_len[b])].copy_(torch.tensor(sent_len[b]))
            t_sent_in_abs[b, :len(sent_in_abs[b])].copy_(
                    torch.tensor(sent_in_abs[b]))

        return t_x, t_sent_len, t_sent_idx, t_sent_in_abs, x_ori_sent

    def get_random_batch_pair(self, batch_size=64, consider_abs_idx=False,
                              data_type='train'):
        """
        Get a random batch of pairs from training set

        Args:
            batch_size:
            consider_abs_idx:

        Returns:
            x           : positive and negative:  [[B x L], [B x L]]
            sent_len    : positive and negative:  [[B], [B]]
            sent_idx    : positive and negative:  [[B], [B]]
        """

        if data_type == 'train':
            data_set = self.emb_train
        elif data_type == 'val':
            data_set = self.emb_val
        elif data_type == 'test':
            data_set = self.emb_test

        _x = []
        _sent_len = []
        _x_sent_ids = []
        _max_len = 0
        id = 0
        while id < batch_size:
            # choose an article first
            item_art = random.choice(data_set)
            # sampling
            sample = self._produce_art_sample(item_art, consider_abs_idx, 1)
            if sample is None:
                continue
            _ , tmp_len, tmp_emb, tmp_idx = sample
            _x.append(tmp_emb)
            _sent_len.append(tmp_len)
            _x_sent_ids.append(tmp_idx)
            if max(*tmp_len) > _max_len:
                _max_len = max(*tmp_len)
            id += 1

        # build tensor
        assert (len(_x) == len(_sent_len) == len(_x_sent_ids))

        # they share the same length vec
        t_x_pos = torch.Tensor(batch_size, _max_len).zero_()
        t_x_neg = torch.Tensor(batch_size, _max_len).zero_()
        t_sent_len_pos = torch.Tensor(batch_size).zero_()
        t_sent_len_neg = torch.Tensor(batch_size).zero_()
        t_sent_idx_pos = torch.Tensor(batch_size).zero_()
        t_sent_idx_neg = torch.Tensor(batch_size).zero_()

        for b in range(batch_size):
            t_x_pos[b, :_x[b][0].size(0)].copy_(_x[b][0])
            t_x_neg[b, :_x[b][1].size(0)].copy_(_x[b][1])

            t_sent_len_pos[b] = _sent_len[b][0]
            t_sent_len_neg[b] = _sent_len[b][1]

            t_sent_idx_pos[b] = _x_sent_ids[b][0]
            t_sent_idx_neg[b] = _x_sent_ids[b][1]

        t_x = torch.cat((t_x_pos[None, :], t_x_neg[None, :]), dim=0)
        t_sent_len = torch.cat((t_sent_len_pos[None, :],
                                t_sent_len_neg[None, :]), dim=0)
        t_sent_idx = torch.cat((t_sent_idx_pos[None, :],
                                t_sent_idx_neg[None, :]), dim=0)

        return t_x, t_sent_len, t_sent_idx

    def get_tgt_res(self, sent_idx, sent_in_abs, top_n=5):
        """
        Args:
            sent_idx: B x N
            sent_in_abs: B x N
            top_n: int

        Return:
            res: list of list
        """
        assert (sent_idx.size(0) == sent_in_abs.size(0))
        assert (sent_idx.size(1) == sent_in_abs.size(1))
        res = []
        B = sent_idx.size(0)
        for b in range(B):
            this_sent_in_abs = sent_in_abs[b].numpy()
            this_sent_idx = sent_idx[b].cpu().numpy()
            indices = numpy.where(this_sent_in_abs != -1)[0]
            elems = this_sent_in_abs[indices]
            indices = indices[elems.argsort()[:top_n]]
            res.append(this_sent_idx[indices].tolist())
        return res

    def _filter(self, x):   # TODO the deleteion is not perfect
        # filter too short sentence
        for i, elem in enumerate(x):
            if self.enable_bert:
                raise NotImplementedError
                sent_len = []
                for ss in elem['sent']:
                    ss = ''.join(ss.strip())
                    ss = self.tokenizer.tokenize(ss)
                    if len(ss) > 512:
                        ss = ss[:512]
                    ss = self.tokenizer.convert_tokens_to_ids(ss)
                    sent_len.append(len(ss))
            else:
                sent_len = [len(sent.strip()) for sent in elem['sent']]
            sent_len = numpy.array(sent_len)
            isent_to_remove = numpy.where(
                    (sent_len == 0) | (sent_len > self.max_sent_len))[0]
            isent_to_remove = sorted(isent_to_remove.tolist(), reverse=True)
            # add an idx to each sentence
            elem['sent_idx'] = list(range(len(elem['sent'])))
            for isent in isent_to_remove:
                del elem['sent'][isent]
                del elem['sent_idx'][isent]
                del elem['sent_in_abs'][isent]

        # filter batch no valid sentence
        for i, elem in enumerate(x):
            if len(elem['sent_idx']) == 0 or\
               len(elem['sent']) == 0 or \
               max(elem['sent_idx']) > self.max_sent_count:
                x[i] = None
        res = list(filter(lambda a: a is not None, x))
        return res

    def _produce_art_sample(self, input_art, is_consider_abs, pair_size):
        assert (len(input_art['sent']) == len(input_art['sent_idx']))
        assert (len(input_art['sent']) == len(input_art['sent_in_abs']))
        assert (len(input_art['sent']) == len(input_art['sent_emb']))

        # first, sample the positive example
        sent_in_abs = numpy.array(input_art['sent_in_abs'])
        cands = numpy.where(sent_in_abs > 0)[0].tolist()
        if len(cands) == 0: return None
        pos = random.choice(cands)

        # then, negative example
        if is_consider_abs:
            cands = numpy.where((sent_in_abs == -1) |\
                        (sent_in_abs > sent_in_abs[pos]))[0].tolist()
        else:
            cands = numpy.where((sent_in_abs == -1))[0].tolist()
        if len(cands) == 0: return None
        neg = random.choice(cands)

        res_sent = (input_art['sent'][pos], input_art['sent'][neg])
        res_len = (len(input_art['sent_emb'][pos]),
                   len(input_art['sent_emb'][neg]))
        res_emb = (torch.tensor(input_art['sent_emb'][pos]),
                   torch.tensor(input_art['sent_emb'][neg]))
        res_idx = (input_art['sent_idx'][pos],
                   input_art['sent_idx'][neg])

        return res_sent, res_len, res_emb, res_idx

    def _split_train_val_test(self, input, rate=0.01):
        random.shuffle(input)
        test_size = math.floor(len(input) * rate)
        val_size = math.floor(len(input)*rate)
        val_input = input[0:val_size]
        test_input = input[val_size:val_size+test_size]
        train_input = input[val_size+test_size:]
        self.logger.info('test-size = {} , val-size = {} , train-size = {}'.\
              format(len(test_input), len(val_input), len(train_input)))
        return train_input, val_input, test_input

    def _add_emb(self, input):
        for i in range(len(input)):
            item = input[i]
            sent = item['sent']
            item['sent_emb'] = []
            for j in range(len(sent)):
                item_sent = sent[j]
                item_sent_tok = item_sent.split(' ')
                item_emb = self.vocab.get_words_idx(item_sent_tok)
                item['sent_emb'].append(item_emb)
            input[i] = item
        return input

    def _add_emb_bert(self, input):
        for i in range(len(input)):
            item = input[i]
            sent = item['sent']
            item['sent_emb'] = []
            for j in range(len(sent)):
                item_sent = sent[j]
                item_sent_com = ''.join(item_sent.strip())
                item_emb = self.tokenizer.tokenize(item_sent_com)
                if len(item_emb) > 512:
                    item_emb = item_emb[:512]
                item_emb = self.tokenizer.convert_tokens_to_ids(item_emb)
                item['sent_emb'].append(item_emb)
            input[i] = item
        return input


if __name__ == '__main__':
    d = DataLoader(all_dir = './prep/all_query_res_nonsorted.pkl')

    this_batch = d.get_iter_batch(batch_size=19)
    t_x, t_sent_len, t_sent_idx, t_sent_in_abs, x_ori_sent = this_batch
    print(t_x.size())
    print(t_sent_len.size())
    print(t_sent_idx.size())
    print(t_sent_in_abs.size())

    this_batch = d.get_random_batch_pair(batch_size=64)
    t_x, t_sent_len, t_sent_idx = this_batch
    print(t_x.size())
    print(t_sent_len.size())
    print(t_sent_idx.size())
