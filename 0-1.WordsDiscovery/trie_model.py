# -*- coding: utf-8 -*-
from collections import defaultdict
import numpy as np

class TrieNode():
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.char = ''
        self.count = 0
        
class Trie():
    '''
    实现如下功能：
    1. 记录总的词频数：total_count
    2. 输入单词，返回其词频：get_freq
    3. 输入单词，返回其子节点的所有char和相应count：get_children_chars
    4. 迭代器返回插入trie的所有单词及其count：get_all_words
    '''
    def __init__(self):
        self.root = TrieNode()
        self.total_count = 0
        
    def insert(self, text):
        node = self.root
        for c in text:
            node = node.children[c]
            node.char = c
        node.count += 1
        self.total_count += 1
        
    def get_all_words(self):
        q = [('', self.root)]
        while q:
            prefix, node = q.pop(0)
            for child in node.children.values():
                if child.count:
                    yield prefix+child.char, child.count
                q.append((prefix+child.char, child))
            
    def get_freq(self, text):
        node = self.root
        for c in text:
            if c not in node.children:
                return 0
            node = node.children[c]
        return node.count
    
    def get_children_chars(self, text):
        node = self.root
        for c in text:
            if c not in node.children:
                return []
            node = node.children[c]
        return [(k.char, k.count) for k in node.children.values()]

if __name__ == '__main__':
    corpus = Trie()
    corpus_inverse = Trie()
    text = '吃葡萄不吐葡萄皮不吃葡萄倒吐葡萄皮'
    words = []
    for i in range(1, 4):
        words += [text[j:j+i] for j in range(len(text)-i+1)]
    print(words)
    for word in words:
        corpus.insert(word)
        corpus_inverse.insert(word[::-1])
    print(f"Freq of 葡萄 is {corpus.get_freq('葡萄')}")
    print(f"Rights chars of 葡萄 is {corpus.get_children_chars('葡萄')}")
    print(f"Left chars of 葡萄 is {corpus_inverse.get_children_chars('萄葡')}")