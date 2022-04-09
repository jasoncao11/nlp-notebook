import csv
import jieba
import re
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

with open("../bert-base-chinese/vocab.txt", encoding="utf-8") as f:
    lines = f.readlines()
char2idx = {}
for index, line in enumerate(lines):
    char2idx[line.strip("\n")] = index
pad_id = char2idx['[PAD]']
cls_id = char2idx['[CLS]']
sep_id = char2idx['[SEP]']
unk_id = char2idx['[UNK]']
mask_id = char2idx['[MASK]']
candidate_ids = list(char2idx.values())
for idx in [pad_id, cls_id, sep_id, unk_id, mask_id]:
    candidate_ids.remove(idx)

def get_new_segment(sent):
    """
    输入一句话: 
    返回一句经过处理的话: 中文全词mask，被分开的词，加上特殊标记("#")，使得后续处理模块，能够知道哪些字是属于同一个词的。
    """
    seq_cws = [token for token in jieba.cut(sent) if token]
    new_segment = []
    for token in seq_cws:
        if len(token) == 1:
            new_segment.append(token)
        else:
            new_segment.append(token[0])
            for c in token[1:]:
                new_segment.append('##' + c)
    return new_segment

def get_training_instances(processed_tok_list, process=True):
    """
    mask机制。
        （1）85%的概率，保留原词不变
        （2）15%的概率，使用以下方式替换
                85%的概率，使用字符'[MASK]'，替换当前token。
                15%的概率，保留原词不变。
    """
    original_token_ids = []
    for t in processed_tok_list:
        if '##' in t:
            original_token_ids.append(char2idx.get(t[2:], unk_id))
        else:
            original_token_ids.append(char2idx.get(t, unk_id)) 
    original_token_ids.insert(0, cls_id)
    original_token_ids.append(sep_id)

    punc = re.compile(r'^[\.\[\],\'\"!?\/\-~～，。！？；;：《》<>、【】「」\{\}“”‘’:\(\)（）]$')  
    length = len(processed_tok_list)
    labels = [-100]*length

    if not process:
        token_ids = original_token_ids[1:-1]
    else:
        #初始化
        token_ids = [mask_id]*length

        rands = np.random.random(length)
        index = -1
        converted_index = set()

        for r, tok in zip(rands, processed_tok_list):
            index += 1
            #print(r, tok, index)
            if index not in converted_index:
                if punc.search(tok):
                    token_ids[index] = char2idx.get(tok, unk_id)
                else:       
                    #初始化起始和终止位置
                    start, end = 0, length
                    if r < 0.15 * 0.85:
                        if '##' in tok:
                            for ind_before in range(index, -1, -1):
                                if '##' not in processed_tok_list[ind_before]:
                                    start = ind_before
                                    break
                            for ind_after in range(start+1, length, 1):
                                if '##' not in processed_tok_list[ind_after]:
                                    end = ind_after
                                    break
                            #print(f'\n mask-start-end:{start}-{end} \n')
                        else:
                            start = index
                            for ind_after in range(start+1, length, 1):
                                if '##' not in processed_tok_list[ind_after]:
                                    end = ind_after
                                    break
                            #print(f'\n mask-start-end:{start}-{end} \n')

                        token_ids[start:end] = [mask_id]*(end-start)
                        for ind in range(start, end):
                            t = processed_tok_list[ind]
                            if '##' in t:
                                labels[ind] = char2idx.get(t[2:], unk_id)
                            else:
                                labels[ind] = char2idx.get(t, unk_id)
                        for ind in range(start, end):
                            converted_index.add(ind)

                    elif r < 0.15:
                        if '##' in tok:
                            for ind_before in range(index, -1, -1):
                                if '##' not in processed_tok_list[ind_before]:
                                    start = ind_before
                                    break
                            for ind_after in range(start+1, length, 1):
                                if '##' not in processed_tok_list[ind_after]:
                                    end = ind_after
                                    break
                            #print(f'\n keep-start-end:{start}-{end} \n')
                        else:
                            start = index
                            for ind_after in range(start+1, length, 1):
                                if '##' not in processed_tok_list[ind_after]:
                                    end = ind_after
                                    break
                            #print(f'\n keep-start-end:{start}-{end} \n')

                        for ind in range(start, end):
                            t = processed_tok_list[ind]
                            if '##' in t:
                                labels[ind] = char2idx.get(t[2:], unk_id)
                                token_ids[ind] = char2idx.get(t[2:], unk_id)
                            else:
                                labels[ind] = char2idx.get(t, unk_id)
                                token_ids[ind] = char2idx.get(t, unk_id)
                        for ind in range(start, end):
                            converted_index.add(ind)              

                    else:
                        if '##' in tok:
                            token_ids[index] = char2idx.get(tok[2:], unk_id)
                        else:
                            token_ids[index] = char2idx.get(tok, unk_id)              

    token_ids.insert(0, cls_id)
    labels.insert(0, -100)
    token_ids.append(sep_id)
    labels.append(-100)
    return original_token_ids, token_ids, labels

def data_generator(batch_size, max_length, min_length_for_mlm, data_file, repeat=1):
    """Generator function that yields batches of data
    """
    cur_original_token_ids = []
    cur_token_ids = []
    cur_labels = []
    cur_mask = []

    with open(data_file, 'r', encoding='utf8') as rf:
        r = csv.reader(rf)  
        for row in r:
            for i in range(repeat):
                text = row[0]
                segment = get_new_segment(text)
                if len(segment) < min_length_for_mlm:
                    original_token_ids, token_ids, labels = get_training_instances(segment, process=False)  
                else:
                    original_token_ids, token_ids, labels = get_training_instances(segment)        

                if len(token_ids) > max_length:
                    original_token_ids = original_token_ids[:max_length-1] + original_token_ids[-1:]
                    token_ids = token_ids[:max_length-1] + token_ids[-1:]
                    labels = labels[:max_length-1] + labels[-1:]
                mask = [1] * len(token_ids)

                cur_original_token_ids.append(torch.tensor(original_token_ids, dtype=torch.long))
                cur_token_ids.append(torch.tensor(token_ids, dtype=torch.long))
                cur_labels.append(torch.tensor(labels, dtype=torch.long))
                cur_mask.append(torch.tensor(mask, dtype=torch.long))

                if len(cur_token_ids) == batch_size:
                    yield {"original_input_ids": pad_sequence(cur_original_token_ids, batch_first=True, padding_value=0),
                        "input_ids": pad_sequence(cur_token_ids, batch_first=True, padding_value=0),
                        "labels": pad_sequence(cur_labels, batch_first=True, padding_value=-100),
                        "attention_mask": pad_sequence(cur_mask, batch_first=True, padding_value=0)}
                    cur_original_token_ids = []
                    cur_token_ids = []
                    cur_labels = []
                    cur_mask = []             

    if cur_token_ids:
        yield {"original_input_ids": pad_sequence(cur_original_token_ids, batch_first=True, padding_value=0),
            "input_ids": pad_sequence(cur_token_ids, batch_first=True, padding_value=0),
            "labels": pad_sequence(cur_labels, batch_first=True, padding_value=-100),
            "attention_mask": pad_sequence(cur_mask, batch_first=True, padding_value=0)}