# -*- coding: utf-8 -*-
import torch
import numpy as np
from sklearn import metrics
from model import TextRCNN_Bert
from load_data import valdataloader

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"

CLS = 2
BERT_PATH = '../bert-base-chinese'

model = TextRCNN_Bert(BERT_PATH, CLS)
model.load_state_dict(torch.load("./saved_model/model_2.pth"))
model.to(device)

model.eval()
predict_all = np.array([], dtype=int)
labels_all = np.array([], dtype=int)        
for tokens_ids, mask, label in valdataloader:
    tokens_ids, mask, label = tokens_ids.to(device), mask.to(device), label.to(device)
    pred = model(tokens_ids, mask)
    pred = torch.max(pred.data, 1)[1].cpu().numpy()
    predict_all = np.append(predict_all, pred)   
    truth = label.cpu().numpy()
    labels_all = np.append(labels_all, truth)    
acc = metrics.accuracy_score(labels_all, predict_all)
print(f'accuracy on dev is {acc}') 