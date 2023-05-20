# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer
from model import Encoder
from load_data import valdataloader

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
device = "cuda" if torch.cuda.is_available() else 'cpu'
model = Encoder.from_pretrained('./saved_model')
model.to(device)
model.eval()
id2label = {1: "B-ORG",
            2: "I-ORG",
            3: "B-PER",
            4: "I-PER",
            5: "B-LOC",
            6: "I-LOC"}


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
    label_ids = batch_data["label_ids"].to(device)

    chars = tokenizer.convert_ids_to_tokens(input_ids[0])
    sent = ''.join(chars)
    print(f"Sent: {sent}")
    labels = [id2label.get(ix.item(), 'o') for ix in label_ids[0]]
    entities = extract(chars, labels)
    gold_num += len(entities)
    print(f'NER: {entities}')

    res = model(input_ids, attention_mask)
    print(res)
    pred_labels = [id2label.get(ix.item(), 'o') for ix in res[0]]
    pred_entities = extract(chars, pred_labels)

    predict_num += len(pred_entities)
    print(f'Predicted NER: {pred_entities}')
    print('---------------\n')

    for pred in pred_entities:
        if pred in entities:
            correct_num += 1

print(f'gold_num = {gold_num}')
print(f'predict_num = {predict_num}')
print(f'correct_num = {correct_num}')
precision = correct_num / predict_num
print(f'precision = {precision}')
recall = correct_num / gold_num
print(f'recall = {recall}')
print(f'f1-score = {2 * precision * recall / (precision + recall)}')