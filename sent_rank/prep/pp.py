import xml.etree.ElementTree
import requests
import re
from pytorch_pretrained_bert import BertTokenizer
import Levenshtein
import pickle as pkl
import math
import sys
import random
import jieba


SHIT_WORDS = [
    '\uf06c',
]


def cut(t):
    t = jieba.cut(t.strip())
    return ' '.join(t)


def clean(t):
    tt = t
    for sw in SHIT_WORDS:
        tt = tt.replace(sw, '')
    return tt


def proc_title(raw_title):
    assert(raw_title.get('name') == 'title')
    text = raw_title.text.strip()
    text = clean(text)
    return text


def proc_abs(raw_abs):
    assert(raw_abs.get('name') == 'abstract')
    abs_list = raw_abs.text.split('\n\u3000\u3000')
    abs_list_str = []
    for elem in abs_list:
        abs_list_str.append(clean(elem.strip()))
    return abs_list_str


def proc_para(raw_para):
    assert(raw_para.get('name') == 'full_text')
    url = raw_para.text
    r = requests.get(url)
    assert(r.status_code == 200)
    sent_list = re.sub('【para:[0-9]*】',
                       ' ',
                       ''.join(r.text.split('\n\n')))\
                  .split('。')
    for i in range(len(sent_list)):
        sent_list[i] = clean(sent_list[i].strip())
    return sent_list


def get_distance(l, r):
    return Levenshtein.ratio(l, r)


def get_top_n_abs2sent(abs_str_list, sent_list):
    score_res = []

    for i in range(len(abs_str_list)):
        abs_str = abs_str_list[i]
        sub_scores = []
        for j in range(len(sent_list)):
            sent = sent_list[j]
            score = get_distance(abs_str, sent)
            tmp = [i, j, score]
            sub_scores.append(tmp)
        score_res.append(sub_scores)

    def take_score(x):
        return x[2]
    sorted_res = []
    for k in range(len(score_res)):
        abs_score = score_res[k]
        tmp_sorted = sorted(abs_score, key=take_score, reverse=True)
        sorted_res.append(tmp_sorted)
    res = []
    for h in range(len(sorted_res)):
        tmp = sorted_res[h]
        tmp = tmp[0]
        res.append(tmp)
    return res



if __name__ == '__main__':

    fileid = sys.argv[1]

    e = xml.etree.ElementTree.parse(f'data_s/output{fileid}.xml').getroot()
    all_data = []

    sample_id = 0

    for row in e.findall('row'):
        element = list(row)
        if len(element) != 3:
            continue

        try:
            item = {}
            title_str = proc_title(element[0])
            abs_list_str = proc_abs(element[1])
            sent_list = proc_para(element[2])

            top_1_res = get_top_n_abs2sent(abs_list_str, sent_list)

            # filter by threshold, less than 0.2
            to_remove_by_threshold = []
            for i, elem in enumerate(top_1_res):
                if elem[2] < 0.2:
                    to_remove_by_threshold.append(i)
            to_remove_by_threshold = sorted(to_remove_by_threshold,
                                            reverse=True)
            for r in to_remove_by_threshold:
                del top_1_res[r]
            if len(top_1_res) == 0:
                continue
                
            # filling in
            item['sent'] = []
            item['sent_in_abs'] = []
            def _find_in_top_res(s):
                for elem in top_1_res:
                    if elem[1] == s:
                        return elem[0]
                return -1
            for _is, sent in enumerate(sent_list):
                item['sent'].append(cut(sent))
                item['sent_in_abs'].append(_find_in_top_res(_is))

            all_data.append(item)

        except AssertionError:
            continue

        sample_id += 1

        if sample_id % 10 == 0:
            print(f'{sample_id} is pp-ed.')

    with open(f'data_s/query_res_{fileid}.pkl', 'wb') as out_fh:
        pkl.dump(all_data, out_fh)
