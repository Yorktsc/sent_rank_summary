from bosonnlp import BosonNLP

nlp = BosonNLP('ESArJA-D.35336.97a5OVDALCAk')
stopwordspath = "/home/jack/sicheng/chinese_stop_words.txt"


def remove_punctuation(text):
    """
    Remove Punctuations among Chinese Documents
    """
    punctuation = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+“”【】 《》"
    text = re.sub(r'[{}]+'.format(punctuation),'',text)
    return text


def remove_stop_words(corpus, stopwordspath):
    """
    Input: a list of token lists, e.x. [["我是","唐思成"],["我","有点","呆"]]
    Output: a list of token lists without stop words
    """
    stopwords = {}.fromkeys([ line.rstrip() for line in open(stopwordspath,"r",encoding='utf-8')])
    for word_list in corpus:
        for word in word_list:
            if word in stopwords:
                word_list.remove(word)
        return corpus
