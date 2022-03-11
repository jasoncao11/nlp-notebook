# -*- coding: utf-8 -*-
import csv
import random as rnd

idx2char = {}
with open("./data/char_vocabs.txt", "r", encoding="utf8") as rf:
    r = csv.reader(rf)
    for ind, line in enumerate(r):
        idx2char[ind] = line[0].strip()
    idx2char[ind+1] = "<UNK>"
char2idx = {char: idx for idx, char in idx2char.items()}

START_TAG, STOP_TAG = "<START>", "<STOP>"
label2idx = {START_TAG: 0, "O": 1, "B-PER": 2, "I-PER": 3, "B-LOC": 4, "I-LOC": 5, "B-ORG": 6, "I-ORG": 7, STOP_TAG: 8}
idx2label = {idx: label for label, idx in label2idx.items()}

all_words = set()
with open("./data/dict.txt", encoding='utf-8') as rf:
    for line in rf:
        word = line.strip().split()[0]
        all_words.add(word)   
print(len(all_words)) 

word2idx = {}
sent = ''
index = -1
with open("./data/train_data", encoding='utf-8') as rf:
    for line in rf:
        if line != '\n':
            char = line.strip().split()[0]
            sent += char
        else:
            length = len(sent)
            for start in range(length):
                for end in range(start+1, length):
                    word = sent[start:end+1]
                    if word in all_words and word not in word2idx:
                        index += 1
                        word2idx[word] = index
            sent = ''

idx2word = {idx: word for word, idx in word2idx.items()}


def data_generator(corpus_path, char2idx, word2idx, label2idx, shuffle=False):
    datas, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            datas.append(sent_)
            labels.append(tag_)
            sent_, tag_ = [], []
    
    num_lines = len(datas)
    lines_index = [*range(num_lines)]
    if shuffle:
        rnd.shuffle(lines_index)

    for index in lines_index:
        chars = datas[index]
        input_ids = [char2idx.get(c, char2idx["<UNK>"]) for c in chars]
        sent = ''.join(chars)
        input_words = []
        length = len(chars)
        for start in range(length):
            temp = []
            for end in range(start+1, length):
                wordid = word2idx.get(sent[start:end+1])
                if wordid:
                    #temp.append([sent[start:end+1], wordid, end+1-start])
                    temp.append([wordid, end+1-start])
            input_words.append(temp) 

        label_ids = [label2idx[label] for label in labels[index]]     
        assert len(input_ids) == len(input_words) == len(label_ids)
        yield sent, input_ids, input_words, label_ids 