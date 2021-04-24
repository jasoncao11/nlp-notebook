# -*- coding: utf-8 -*-
import torch
from model import BiLSTM_CRF_PARALLEL
from settings import EMBEDDING_DIM, HIDDEN_DIM, TEST_DATA_PATH
from load_data import vocab2idx, idx2vocab, label2idx, idx2label, data_generator

device = "cuda" if torch.cuda.is_available() else 'cpu'

model = BiLSTM_CRF_PARALLEL(len(vocab2idx), label2idx, EMBEDDING_DIM, HIDDEN_DIM).to(device)
model.load_state_dict(torch.load("./saved_model/model.pth"))
model.eval()

def predict(sent, tags):
    result = []
    pre = ''
    w = []
    for idx, tag in enumerate(tags):
        if not pre:
            if tag.startswith('B'):
                pre = tag.split('-')[1]
                w.append(sent[idx])
        else:
            if tag == f'I-{pre}':
                w.append(sent[idx])
            else:
                result.append([w, pre])
                w = []
                pre = ''
                if tag.startswith('B'):
                    pre = tag.split('-')[1]
                    w.append(sent[idx])      
    return [[''.join(x[0]), x[1]] for x in result]

gold_num = 0
predict_num = 0
correct_num = 0

for sentence, tags in data_generator(TEST_DATA_PATH, vocab2idx, label2idx):
    for sent_, tag_ in zip(sentence, tags):
        sent = [idx2vocab[ix.item()] for ix in sent_]
        print (f"Sent: {''.join(sent).replace('<PAD>', '')}")
        tags = [idx2label[ix.item()] for ix in tag_]
        ner = predict(sent, tags)
        gold_num += len(ner)
        
        print (f'NER: {ner}')       
        pred = model(sent_.unsqueeze(0).to(device))
        pre_tags = [idx2label[ix] for ix in pred[0][1]]
        
        pred_ner = predict(sent, pre_tags)
        predict_num += len(pred_ner)
        print (f'Predicted NER: {pred_ner}')
        print ('---------------\n')
        
        for pred in pred_ner:
            if pred in ner:
                correct_num += 1

print(f'gold_num = {gold_num}')
print(f'predict_num = {predict_num}')
print(f'correct_num = {correct_num}')
precision = correct_num/predict_num
print(f'precision = {precision}')
recall = correct_num/gold_num
print(f'recall = {recall}')
print(f'f1-score = {2*precision*recall/(precision+recall)}')