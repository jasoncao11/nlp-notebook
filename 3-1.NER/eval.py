# -*- coding: utf-8 -*-
import torch
from model import BiLSTM_CRF
from load_data import vocab2idx, idx2vocab, label2idx, idx2label, data_generator

device = "cuda" if torch.cuda.is_available() else 'cpu'

EMBEDDING_DIM = 300
HIDDEN_DIM = 64
BATCH_SIZE = 1
TEST_DATA_PATH = "./data/test_data" # 测试数据

model = BiLSTM_CRF(len(vocab2idx), label2idx, EMBEDDING_DIM, HIDDEN_DIM).to(device)
model.load_state_dict(torch.load("./saved_model/model.pth", map_location=device))
model.eval()

def extract(sent, tags):
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

for inputs_idx_batch, labels_idx_batch, real_lengths in data_generator(TEST_DATA_PATH, vocab2idx, label2idx, BATCH_SIZE):
    print(inputs_idx_batch)
    print(labels_idx_batch)
    if len(inputs_idx_batch) > 0:
        sent = [idx2vocab[ix.item()] for ix in inputs_idx_batch[0]]
        print(f"Sent: {''.join(sent)}")
        labels = [idx2label[ix.item()] for ix in labels_idx_batch[0]]
        entities = extract(sent, labels)
        gold_num += len(entities)
        print (f'NER: {entities}')

        res = model(inputs_idx_batch.to(device))
        pred_labels = [idx2label[ix] for ix in res[1]]
        
        pred_entities = extract(sent, pred_labels)
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