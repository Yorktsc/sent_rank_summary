# coding = utf-8

# @time    : 2019/7/19 11:16 PM
# @author  : alchemistlee
# @fileName: infer.py
# @abstract:


import os
import torch
from utils import *
import jieba


def cut(t):
    t = jieba.cut(t.strip())
    return ' '.join(t)


class Tester(object):
    def __init__(self, opt, model, dataloader):
        self.opt = opt
        self.model = model
        tmp_model_name = '{}/model_best.pt'.format(opt.outf)
        assert(os.path.exists(tmp_model_name))
        self.load_trained_model(tmp_model_name)

        self.dataloader = dataloader

    def load_trained_model(self, path):
        '''
        Args:
            path: path/to/model
        '''
        loaded = torch.load(path)
        self.model.load_state_dict(loaded)

    def inference_example(self):
        '''
        main function
        '''

        B = 4
        this_batch = self.dataloader.get_iter_batch(
                batch_size=B, data_type='val')
        x, x_len, sent_idx, sent_in_abs, input_ori = this_batch
        x = x.long().to('cuda')
        x_len = x_len.long().to('cuda')
        sent_idx = sent_idx.long().to('cuda')

        pred_res = self.model.inference(x, x_len, sent_idx,
                                        top_n=self.opt.top_n)
        for b in range(B):
            print(f'------------------ {b} ------------------')
            print_ori_sent_full(input_ori[b], pred_res[b].tolist())

    def inference_example_sent(self, filename='testsina.txt'):
        # read file
        with open(filename, 'r') as fb:
            content = fb.readlines()
            content = list(filter(lambda x: len(x.strip()) > 0, content))
            content = ''.join(content)

        # split pp
        content = content.split('。')
        items = {'sent': []}
        for elem in content:
            items['sent'].append(cut(elem))
        items['sent_in_abs'] = [-1 for _ in range(len(content))]
        items_lst = self.dataloader._filter([items])
        items_lst_with_emb = self.dataloader._add_emb(items_lst)
        this_batch = self.dataloader.get_iter_batch(
                batch_size=1, data_type=items_lst_with_emb)
        x, x_len, sent_idx, sent_in_abs, input_ori = this_batch
        x = x.long().to('cuda')
        x_len = x_len.long().to('cuda')
        sent_idx = sent_idx.long().to('cuda')
        
        pred_res = self.model.inference(x, x_len, sent_idx,
                                        top_n=self.opt.top_n)
        
        print_ori_sent_full(input_ori[0], pred_res[0].tolist())
