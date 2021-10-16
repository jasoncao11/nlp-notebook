# -*- coding: utf-8 -*-
import torch
from model import LatticeLSTM
from load_data import char2idx, idx2char, label2idx, idx2label, word2idx, data_generator

character_size = len(char2idx)
word_size = len(word2idx)
embed_dim = 300
hidden_dim = 128

TEST_DATA_PATH = "./data/test_data" # 测试数据
device = "cuda" if torch.cuda.is_available() else 'cpu'

model = LatticeLSTM(character_size, word_size, label2idx, embed_dim, hidden_dim).to(device)
model.load_state_dict(torch.load("./saved_model/model_lattice.pth", map_location=device))
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

for sent, input_ids, input_words, labels_idx in data_generator(TEST_DATA_PATH, char2idx, word2idx, label2idx):
    print(f"Sent: {sent}")
    chars = [idx2char[ix] for ix in input_ids]
    labels = [idx2label[ix] for ix in labels_idx]
    entities = extract(chars, labels)
    gold_num += len(entities)
    print (f'NER: {entities}')

    res = model(input_ids, input_words)
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