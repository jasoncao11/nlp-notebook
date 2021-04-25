# -*- coding: utf-8 -*-
import re
import numpy as np
from collections import defaultdict
from trie_model import Trie

RE_PUNCTUATION_TO_CLEAN = r'[.:;?!\~,\-_()[\]<>。：；？！~，、——（）【】《》＃＊＝＋/｜‘’“”￥#*=+\\|\'"^$%`]'

class NewWords():
    def __init__(self):
        self.trie = Trie()
        self.trie_reversed = Trie()
        self.word_info = defaultdict(dict)
        self.NGRAM = 4
        self.WORD_MIN_LEN = 2
        self.WORD_MIN_FREQ = 2
        self.WORD_MIN_PMI = 8
        self.WORD_MIN_NEIGHBOR_ENTROPY = 1

    def parse(self, text):
        words = self.n_gram_words(text)
        self.build_trie(words)
        self.get_words_pmi()
        self.get_words_entropy()

    def n_gram_words(self, text):
        text = re.sub(RE_PUNCTUATION_TO_CLEAN, '', text)
        words = []
        for i in range(1, self.NGRAM+1):
            words += [text[j:j+i] for j in range(len(text)-i+1)]   
        return words

    def build_trie(self, words):
        for word in words:
            self.trie.insert(word)
            self.trie_reversed.insert(word[::-1])
            
    def get_words_pmi(self):
        for word, count in self.trie.get_all_words():
            if len(word) < self.WORD_MIN_LEN or count < self.WORD_MIN_FREQ:
                continue
            pmi = min([count * self.trie.total_count / self.trie.get_freq(word[:i]) / self.trie.get_freq(word[i:]) for i in range(1, len(word))\
                       if self.trie.get_freq(word[:i]) and self.trie.get_freq(word[i:])])
            pmi = np.log2(pmi)
            self.word_info[word]['pmi'] = pmi
            self.word_info[word]['freq'] = count
                
    def calculate_entropy(self, char_list):
        if not char_list:
            return 0
        num = sum([v for k, v in char_list])
        entropy = (-1)*sum([(v/num)*np.log2(v/num) for k, v in char_list])
        return entropy

    def get_words_entropy(self):
        for k, v in self.word_info.items():
            
            right_char = self.trie.get_children_chars(k)
            right_entropy = self.calculate_entropy(right_char)

            left_char = self.trie_reversed.get_children_chars(k[::-1])
            left_entropy = self.calculate_entropy(left_char)
            
            entropy = min(right_entropy, left_entropy)
            self.word_info[k]['entropy'] = entropy
            
    def candidates(self, sortby='pmi'):
        res = [(k, v) for k, v in self.word_info.items() if v['pmi'] >= self.WORD_MIN_PMI and v['entropy'] >= self.WORD_MIN_NEIGHBOR_ENTROPY]
        res = sorted(res, key=lambda x: x[1][sortby], reverse=True)
        for k, v in res:
            yield k, v
                       
if __name__ == '__main__':
    discover = NewWords()
    discover.parse('''中国科兴生物研发的克尔来福是一种灭活疫苗，由已杀灭的病原体制成，主要通过其中的抗原诱导细胞免疫的产生。另外几种疫苗，例如莫德纳和辉瑞的疫苗都属于核糖核酸疫苗，使用的是RNA疫苗原理，抽取病毒内部分核糖核酸编码蛋白制成疫苗。新加坡南洋理工大学感染与免疫副教授罗大海对BBC表示，“克尔来福是用比较传统的方法制成的（灭活）疫苗，灭活疫苗使用广泛而且非常成功，例如狂犬病疫苗。”理论上，科兴疫苗主要的优势在于它能够在常规冰箱温度下（2至8摄氏度）保存，这一点和牛津/阿斯利康研发的病毒载体疫苗有相同优点。莫德纳的疫苗必须存放在摄氏零下20度，而辉瑞疫苗必须存放在摄氏零下70度。这意味着科兴和牛津/阿斯利康这两种疫苗，更能有效地在发展中国家使用，因为那些地方可能没有足够的低温储存设备供疫苗保存。但是，相对于最新加入接种行列的单剂疫苗 — 美国杨森和中国康希诺 — 而言，科兴疫苗仍需注射两针。疫苗谣言的打破：改变DNA、植入微芯片等疫苗阴谋论。新冠疫苗接种在即，你该了解的四大问题。效果如何？科兴疫苗三期临床试验在4个国家展开，各国试验结果相差较大，有效性从50% - 90%不等。从2021年1月以来，至少有7个国家先后批准科兴疫苗紧急使用。不过到目前为止它的三期临床整体有效性数据仍未公布。截止今年3月8日，香港有10多万人接种第一剂科兴疫苗，虽然近期出现三宗接种科兴疫苗后死亡的案例，但港府新冠疫苗临床事件评估专家委员会对三宗案例的调查结果称科兴疫苗与死亡并无直接关系。今年1月13日，科兴董事长在谷物元联防联控机制发布会上给出一组数据：土耳其中期分析结果显示该疫苗保护率91.25%；印尼三期临床试验保护率65.3%；巴西三期临床试验从2020年10月开始，试验结果显示重症保护率达100%，对高危人群总体保护率达50.3%。''')
    for k, v in discover.candidates():
        print(k, v)   