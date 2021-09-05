# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn import metrics
from model import TextCNN
from load_data import traindataloader, valdataloader, char2id

EPOCHS = 10
CLS = 2
device = "cuda" if torch.cuda.is_available() else 'cpu'

def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
    loss = (p_loss + q_loss) / 2
    return loss

def train():

    model = TextCNN(len(char2id), CLS)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss= []
        for batch in traindataloader:           
            text_idx_batch, label_idx_batch = batch['input_ids'].to(device), batch['labels'].to(device)
            model.zero_grad()
            out = model(text_idx_batch)

            ce_loss = criterion(out, label_idx_batch)
            kl_loss = compute_kl_loss(out[::2], out[1::2])
            loss = ce_loss + 4 * kl_loss

            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()   
        print(f'Epoch[{epoch}] - Loss:{np.mean(epoch_loss)}')

        model.eval()
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():        
            for batch in valdataloader:
                text_idx_batch, label_idx_batch = batch['input_ids'].to(device), batch['labels']
                pred = model(text_idx_batch)
                pred = torch.max(pred.data, 1)[1].cpu().numpy()
                predict_all = np.append(predict_all, pred)
                
                truth = label_idx_batch.cpu().numpy()
                labels_all = np.append(labels_all, truth)            
            
        acc = metrics.accuracy_score(labels_all, predict_all)
        print(f'Epoch[{epoch}] - acc:{acc}')

if __name__ == '__main__':
    train()