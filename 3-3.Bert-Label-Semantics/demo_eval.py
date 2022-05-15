# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizer
from model import Label_Encoder, Token_Encoder
from load_data import valdataloader, idx2label
from load_label import label_input_ids, label_attention_mask

tokenizer = BertTokenizer.from_pretrained('./bert-base-chinese')
device = "cuda" if torch.cuda.is_available() else 'cpu'

label_encoder = Label_Encoder.from_pretrained('./saved_model_label_encoder')
label_encoder.to(device)
label_encoder.eval()

token_encoder = Token_Encoder.from_pretrained('./saved_model_token_encoder')
token_encoder.to(device)
token_encoder.eval()

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

    token_input_ids = batch_data["input_ids"].to(device)
    token_attention_mask = batch_data["attention_mask"].to(device)
    label_ids = batch_data["label_ids"].to(device)

    chars = tokenizer.convert_ids_to_tokens(token_input_ids[0][1:-1])
    sent = ''.join(chars)
    print(f"Sent: {sent}")
    labels = [idx2label[ix.item()] for ix in label_ids[0][1:-1]]
    entities = extract(chars, labels)
    gold_num += len(entities)
    print (f'NER: {entities}')

    batch_size = token_input_ids.shape[0]
    label_logits = label_encoder(label_input_ids, label_attention_mask)
    label_logits = label_logits.transpose(1,0).repeat(batch_size,1,1) # 1(batch size)*768*7(number of labels)
    token_logits = token_encoder(token_input_ids, token_attention_mask)# 1*seq*768
    logits = torch.matmul(token_logits, label_logits) # 1*seq*7
    pred = torch.argmax(logits, dim=-1)  
    
    pred_labels = [idx2label[ix.item()] for ix in pred[0][1:-1]]
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