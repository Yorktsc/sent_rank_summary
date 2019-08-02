# coding = utf-8

# @time    : 2019/7/17 2:46 PM
# @author  : alchemistlee
# @fileName: train.py
# @abstract:


import torch
from utils import *


class Trainer(object):
    def __init__(self, opt, model, dataloader, optimizer, logger):
        self.opt = opt
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer

        self.best_eval_f1 = 0
        self.logger = logger

    def launch(self):
        opt = self.opt
        model = self.model
        dataloader = self.dataloader
        optimizer = self.optimizer

        train_loss, n_train = 0, 0
        lr_decay_count = 0

        for iter in range(opt.max_niter):
            model.train()

            optimizer.zero_grad()
            x, x_len, sent_idx = dataloader.get_random_batch_pair(
                    batch_size=opt.batch_size, data_type='train')
            x = x.long().to('cuda')
            sent_idx = sent_idx.long().to('cuda')
            x_len = x_len.long().to('cuda')

            loss = model.compute_loss_pair(x, x_len, sent_idx)
            loss.backward()

            train_loss += loss.item() * x.size(1)
            n_train += x.size(1)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_norm)
            optimizer.step()

            if iter % opt.eval_niter == 0:
                self.logger.info(f'iter = {iter} ,  train loss = {train_loss/n_train}')
                cur_eval_f1, best_eval_f1, lr_decay_count =\
                        self.eval_and_adjust(lr_decay_count)
                self.logger.info('iter = {} , best-f1 = {} , cur-f1 = {}'.\
                      format(iter, best_eval_f1, cur_eval_f1))
                self.best_eval_f1 = best_eval_f1

    def eval_and_adjust(self, lr_decay_count=None):
        opt = self.opt
        model = self.model
        dataloader = self.dataloader
        optimizer = self.optimizer
        best_eval_f1 = self.best_eval_f1
        
        model.eval()
        n_eval, eval_f1 = 0, 0
        id = 0
        with torch.no_grad():
            while True:
                this_batch = dataloader.get_iter_batch(
                        batch_size=opt.eval_batch_size, data_type='val')
                if this_batch is None:
                    break
                x, x_len, sent_idx, sent_in_abs, input_ori = this_batch
                x = x.long().to('cuda')
                x_len = x_len.long().to('cuda')
                sent_idx = sent_idx.long().to('cuda')
                # inference
                tgt_res = dataloader.get_tgt_res(sent_idx, sent_in_abs, opt.top_n)
                pred_res = model.inference(x, x_len, sent_idx, top_n=opt.top_n)
                score_res = self._evaluate_f1(pred_res, tgt_res, input_ori,
                                        print_sent=id % 100 == 0)
                n_eval += x.size(0)
                eval_f1 += sum(score_res)
                id += 1
            cur_eval_f1 = eval_f1 / n_eval

            # adjusting
            if lr_decay_count is not None:
                if cur_eval_f1 > best_eval_f1:
                    lr_decay_count = 0
                    best_eval_f1 = cur_eval_f1
                    tmp_model_name = '{}/model_best.pt'.format(opt.outf)
                    self.logger.info('saving model , {} '.format(tmp_model_name))
                    torch.save(model.state_dict(), tmp_model_name)
                else:
                    lr_decay_count += 1
                    if lr_decay_count == opt.lr_decay_pat:
                        self.logger.info('learning rate decay ... ')
                        for p in optimizer.param_groups:
                            p['lr'] /= 2
                            cur_lr = p['lr']
                        lr_decay_count = 0
                        if cur_lr < 1e-6:
                            self.logger.info('learning rate stop iter !'
                                  'best-acc = {} , cur-acc = {}'.\
                                  format(best_eval_f1, cur_eval_f1))
                            raise KeyboardInterrupt   # TODO ugly
                        else:
                            self.logger.info('learning rate now: {}'.format(cur_lr))
                return cur_eval_f1, best_eval_f1, lr_decay_count
            else:
                return cur_eval_f1

    @staticmethod
    def _evaluate_f1(pred_res, tgt_res, input_ori, print_sent=False):
        res = []
        assert (len(pred_res) == len(tgt_res))

        for b in range(len(pred_res)):
            item_pred = pred_res[b]
            item_tgt = tgt_res[b]
            item_pred_set, item_tgt_set = set(item_pred), set(item_tgt)

            itersect = item_pred_set.intersection(item_tgt_set)
            item_pred_diff = item_pred_set.difference(item_tgt_set)
            item_tgt_diff = item_tgt_set.difference(item_pred_set)

            precision, recall = 0, 0
            if len(item_pred_set) == 0 or len(item_tgt_set) == 0:
                continue
            else:
                precision = len(itersect) / len(item_pred_set)
                recall = len(itersect) / len(item_tgt_set)

            if precision + recall == 0:
                res.append(0)
            else:
                res.append(2 * (precision * recall) / (precision + recall))

        '''
        self.logger.info('pred = {}'.format(item_pred))
        self.logger.info('tgt = {}'.format(item_tgt))
        '''
        # print out the last item in this batch
        if print_sent:
            print('diff : ')
            print_diff_sent(list(item_pred_diff), list(item_tgt_diff), input_ori[-1])
            print('intersect : ')
            print_ori_sent(input_ori[-1], list(itersect))

        return res
