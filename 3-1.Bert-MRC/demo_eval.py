# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer
from model import MRCModel
from load_data import TEST_DATA_PATH, template, MAX_LEN

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
device = "cuda" if torch.cuda.is_available() else 'cpu'
model = MRCModel.from_pretrained('./saved_model')
model.to(device)
model.eval()

def extract(chars, tags):
    result = []
    pre = ''
    w = []
    for idx, tag in enumerate(tags):
        if not pre:
            if tag.startswith('B'):
                pre = tag.split('-')[1]
                w.append(chars[idx])
        else:
            if tag == f'I-{pre}':
                w.append(chars[idx])
            else:
                result.append([w, pre])
                w = []
                pre = ''
                if tag.startswith('B'):
                    pre = tag.split('-')[1]
                    w.append(chars[idx])      
    return [[''.join(x[0]), x[1]] for x in result]

def mrc_decode(start_pred, end_pred, raw_text):
    predict_entities = []
    for i, s_type in enumerate(start_pred):
        if s_type == 0:
            continue
        for j, e_type in enumerate(end_pred[i:]):
            if s_type == e_type:
                tmp_ent = raw_text[i:i+j+1]
                predict_entities.append(tmp_ent)
                break
    return predict_entities

gold_num = 0
predict_num = 0
correct_num = 0
with open (TEST_DATA_PATH, encoding='utf8') as rf:
    chars = []
    labels = []
    origin_labels = []
    for line in rf:
        if line != '\n':
            char, label = line.strip().split()
            chars.append(char)
            origin_labels.append(label)
            if '-' in label:
                label = label.split('-')[1]
            labels.append(label)
        else:  
            sent = ''.join(chars)
            print(f"Sent: {sent}")
            entities = extract(chars, origin_labels)
            gold_num += len(entities)
            print (f'NER: {entities}')            
            
            pred_entities = []
            for prefix, target in template:                       
                input_ids_1 = [tokenizer.convert_tokens_to_ids(c) for c in prefix]
                input_ids_1 = [tokenizer.cls_token_id] + input_ids_1 + [tokenizer.sep_token_id]
                token_type_ids_1 = [0] * len(input_ids_1)                    
                if len(chars)+1+len(input_ids_1) > MAX_LEN:
                    chars = chars[:MAX_LEN-1-len(input_ids_1)]

                input_ids_2 = [tokenizer.convert_tokens_to_ids(c) for c in chars]
                input_ids_2 = input_ids_2 + [tokenizer.sep_token_id]
                token_type_ids_2 = [1] * len(input_ids_2)
                           
                input_ids = input_ids_1 + input_ids_2
                input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
                token_type_ids = token_type_ids_1 + token_type_ids_2
                token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(device)
                attention_mask = [1]*len(input_ids)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)
                
                start_pred, end_pred = model(input_ids, attention_mask, token_type_ids)
                start_pred = start_pred.squeeze(0)[len(input_ids_1):-1]
                end_pred = end_pred.squeeze(0)[len(input_ids_1):-1]
                predict_entities = mrc_decode(start_pred, end_pred, sent)
                for pred in predict_entities:
                    pred_entities.append([pred, target])
            chars = []
            labels = []
            origin_labels = []
    
            predict_num += len(pred_entities)
            print (f'Predicted NER: {pred_entities}')
            print ('---------------\n')
            for pred in pred_entities:
                if pred in entities:
                    correct_num += 1          

print(f'gold_num = {gold_num}')
print(f'predict_num = {predict_num}')
print(f'correct_num = {correct_num}')
precision = correct_num/predict_num
print(f'precision = {precision}')
recall = correct_num/gold_num
print(f'recall = {recall}')
print(f'f1-score = {2*precision*recall/(precision+recall)}')