# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer
from model import BertForNER
from load_data import valdataloader, idx2label

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
device = "cuda" if torch.cuda.is_available() else 'cpu'
model = BertForNER.from_pretrained('./saved_model')
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

gold_num = 0
predict_num = 0
correct_num = 0

for batch_data in valdataloader:

    input_ids = batch_data["input_ids"].to(device)
    attention_mask = batch_data["attention_mask"].to(device)
    labels_idx = batch_data["labels_idx"].to(device)

    chars = tokenizer.convert_ids_to_tokens(input_ids[0])
    sent = ''.join(chars)
    print(f"Sent: {sent}")
    labels = [idx2label[ix.item()] for ix in labels_idx[0]]
    entities = extract(chars, labels)
    gold_num += len(entities)
    print (f'NER: {entities}')

    res = model(input_ids, attention_mask)
    pred_labels = [idx2label[ix] for ix in res[1]]
    pred_entities = extract(chars, pred_labels)
    
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