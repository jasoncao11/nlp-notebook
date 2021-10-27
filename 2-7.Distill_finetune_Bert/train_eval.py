# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from model import TextRCNN_Bert
from load_data import traindataloader, valdataloader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

SAVED_DIR = './saved_model'
EPOCHS = 5
BERT_PATH = './bert-base-chinese'
WARMUP_PROPORTION = 0.1
device = "cuda" if torch.cuda.is_available() else 'cpu'

model = TextRCNN_Bert.from_pretrained(BERT_PATH)
model.to(device)

total_steps = len(traindataloader) * EPOCHS
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps), num_training_steps=total_steps)
criterion = nn.NLLLoss()

loss_vals = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss= []
    pbar = tqdm(traindataloader)
    pbar.set_description("[Epoch {}]".format(epoch)) 
    for tokens_ids, mask, label in pbar:
        tokens_ids, mask, label = tokens_ids.to(device), mask.to(device), label.to(device)
        model.zero_grad()
        out = model(tokens_ids, mask)
        loss = criterion(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        epoch_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
    loss_vals.append(np.mean(epoch_loss))    
    
model.save_pretrained(SAVED_DIR)
plt.plot(np.linspace(1, EPOCHS, EPOCHS).astype(int), loss_vals)

model.eval()
predict_all = np.array([], dtype=int)
labels_all = np.array([], dtype=int)
with torch.no_grad():        
    for tokens_ids, mask, label in valdataloader:
        tokens_ids, mask, label = tokens_ids.to(device), mask.to(device), label.to(device)
        pred = model(tokens_ids, mask)
        pred = torch.max(pred.data, 1)[1].cpu().numpy()
        predict_all = np.append(predict_all, pred)   
        truth = label.cpu().numpy()
        labels_all = np.append(labels_all, truth)    
acc = metrics.accuracy_score(labels_all, predict_all)
print(f'accuracy on dev is {acc}')