# -*- coding: utf-8 -*-
import unicodedata

class Tokenizer():

    with open("./bert-base-chinese/vocab.txt", encoding="utf-8") as f:
        lines = f.readlines()
    word2idx = {}
    for index, line in enumerate(lines):
        word2idx[line.strip("\n")] = index
    cls_id = word2idx['[CLS]']
    sep_id = word2idx['[SEP]']
    unk_id = word2idx['[UNK]']
    idx2word = {idx: word for word, idx in word2idx.items()}

    @classmethod
    def encode(cls, first_text, second_text=None, max_length=512):
        first_text = first_text.lower()
        first_text = unicodedata.normalize('NFD', first_text)

        first_token_ids = [cls.word2idx.get(t, cls.unk_id) for t in first_text]
        first_token_ids.insert(0, cls.cls_id)
        first_token_ids.append(cls.sep_id)

        if second_text:
            second_text = second_text.lower()
            second_text = unicodedata.normalize('NFD', second_text)

            second_token_ids = [cls.word2idx.get(t, cls.unk_id) for t in second_text]
            second_token_ids.append(cls.sep_id)
        else:
            second_token_ids = []

        while True:
            total_length = len(first_token_ids) + len(second_token_ids)
            if total_length <= max_length:
                break
            elif len(first_token_ids) > len(second_token_ids):
                first_token_ids.pop(-2)
            else:
                second_token_ids.pop(-2)

        first_token_type_ids = [0] * len(first_token_ids)
        first_token_type_ids_for_mask = [1] * len(first_token_ids)
        labels = [-100] * len(first_token_ids) 

        if second_token_ids:
            second_token_type_ids = [1] * len(second_token_ids)
            second_token_type_ids_for_mask = [0] * len(second_token_ids)

            first_token_ids.extend(second_token_ids)
            first_token_type_ids.extend(second_token_type_ids)
            first_token_type_ids_for_mask.extend(second_token_type_ids_for_mask)
            labels.extend(second_token_ids)
            
        return first_token_ids, first_token_type_ids, first_token_type_ids_for_mask, labels

    @classmethod
    def decode(cls, input_ids):
      tokens = [cls.idx2word[idx] for idx in input_ids]
      return ' '.join(tokens)