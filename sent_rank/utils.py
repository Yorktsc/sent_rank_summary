__all__  = ['print_ori_sent', 'print_diff_sent', 'print_ori_sent_full']

import pprint
import re


class bcolors:
    KEY = '\033[95m'
    ENDC = '\033[0m'
COLORS = [bcolors.KEY]


def print_ori_sent(input_ori, input_idx):
    res = []
    try:
        for i in range(len(input_idx)):
            idx = input_idx[i]
            res.append(input_ori[idx])
    except Exception as ex:
        print('--- print ori ---')
        print(input_ori)
        print(input_idx)
        raise ex
    print(res)


def print_ori_sent_full(input_ori, input_idx):
    for k, elem in input_ori.items():
        if k in input_idx:
            print(COLORS[0] + elem.strip() + COLORS[0] + bcolors.ENDC)
        else:
            print(elem.strip())


def adjust_cut_lst(input_lst, adjust_len, width):
    res = []
    for i in range(adjust_len):
        if i < len(input_lst):
            if len(input_lst[i]) < width:
                input_lst[i] = input_lst[i].ljust(width,' ')
            res.append(input_lst[i])
        else:
            res.append(' '.ljust(width, ' '))
    return res


def format_2_col(l_input, r_input, width=50):

    def cut_text(text, lenth):
        textArr = re.findall('.{'+str(lenth)+'}', text)
        textArr.append(text[(len(textArr)*lenth):])
        return textArr

    l_cut_lst = cut_text(l_input, width)
    r_cut_lst = cut_text(r_input, width)
    max_len_cut_lst = len(l_cut_lst) if len(l_cut_lst) > len(r_cut_lst) else len(r_cut_lst)

    l_proc_res = adjust_cut_lst(l_cut_lst, max_len_cut_lst, width)
    r_proc_res = adjust_cut_lst(r_cut_lst, max_len_cut_lst, width)

    return l_proc_res, r_proc_res


def print_diff_sent(pred_idx, tgt_idx, input_ori, is_sort=True):
    if is_sort:
        pred_idx.sort()
        tgt_idx.sort()
    res = []
    pp = pprint.PrettyPrinter(width=200)

    max_len = len(pred_idx) if len(pred_idx)>len(tgt_idx) else len(tgt_idx)
    for i in range(max_len):
        pred_item = 'None'
        tgt_item = 'None'
        pred_item_idx = '-1'
        tgt_item_idx = '-1'
        if i < len(pred_idx):
            pred_item_idx = pred_idx[i]
            pred_item = input_ori[pred_idx[i]]
        if i < len(tgt_idx):
            tgt_item_idx = tgt_idx[i]
            tgt_item = input_ori[tgt_idx[i]]
        pred_proc, tgt_proc = format_2_col(pred_item, tgt_item)

        assert (len(pred_proc) == len(tgt_proc))
        for j in range(len(pred_proc)):
            if j == 0:
                tmp = '[{}] : {} ||| [{}] : {}'.format(pred_item_idx,
                       pred_proc[j], tgt_item_idx, tgt_proc[j])
            else:
                tmp = '  {} ||| {}'.format(pred_proc[j], tgt_proc[j])
            res.append(tmp)
    pp.pprint(res)

