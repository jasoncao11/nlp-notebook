# -*- coding: utf-8 -*-
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn import metrics
from model import TextRCNN_Bert
from load_data import traindataloader, valdataloader
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else 'cpu'  
torch.cuda.empty_cache()

class TextRCNN(nn.Module):
    def __init__(self, class_num=2, vocab_size=21128):
        super(TextRCNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, 200)        
        self.lstm = nn.LSTM(200, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2 + 200, class_num)

    def forward(self, x, softmax=True):
        embeds = self.embed(x)  # [batch_size, seq_len, embed_dim]
        out, _ = self.lstm(embeds) # [batch_size, seq_len, hidden_size * 2]
        out = torch.cat((embeds, out), 2) # [batch_size, seq_len, hidden_size * 2 + embed_dim]
        out = F.relu(out) # [batch_size, seq_len, hidden_size * 2 + embed_dim]
        out = out.permute(0, 2, 1) # [batch_size, hidden_size * 2 + embed_dim, seq_len]
        out = F.max_pool1d(out, out.size(2)).squeeze(2) # [batch_size, hidden_size * 2 + embed_dim]
        if softmax:         
            logit = F.log_softmax(self.fc(out), dim=1) # [batch_size, class_num]
        else:
            logit = self.fc(out) # [batch_size, class_num]
        return logit

def kd_step(saved_model, temperature, epochs, traindata, valdata, class_num=2):
    
    teacher = TextRCNN_Bert.from_pretrained(saved_model)
    teacher.to(device)
    teacher.eval()
    
    student = TextRCNN()
    student.to(device)
    student.train()    
    optimizer = optim.Adam(student.parameters(), lr=0.0001)
    KD_loss = nn.KLDivLoss(reduction='batchmean')
    #train
    loss_vals = []
    for epoch in range(epochs):
        epoch_loss= []
        pbar = tqdm(traindata)
        pbar.set_description("[Epoch {}]".format(epoch)) 
        for tokens_ids, mask, label in pbar:
            tokens_ids, mask, label = tokens_ids.to(device), mask.to(device), label.to(device)
            with torch.no_grad():
                logits_t = teacher(tokens_ids, mask, softmax=False)
            student.zero_grad()            
            logits_s = student(tokens_ids, softmax=False)
            loss = KD_loss(input=F.log_softmax(logits_s/temperature, dim=-1),
                           target=F.softmax(logits_t/temperature, dim=-1))            
            loss.backward()
            epoch_loss.append(loss.item())
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
        loss_vals.append(np.mean(epoch_loss))
    plt.plot(np.linspace(1, epochs, epochs).astype(int), loss_vals)    
    #eval
    student.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():         
        for tokens_ids, mask, label in valdata:
            tokens_ids, mask, label = tokens_ids.to(device), mask.to(device), label.to(device)
            pred = student(tokens_ids)
            pred = torch.max(pred.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, pred)   
            truth = label.cpu().numpy()
            labels_all = np.append(labels_all, truth)        
    acc = metrics.accuracy_score(labels_all, predict_all)
    print(acc)

if __name__ == '__main__':
    kd_step('./saved_model',            
            5, 
            5, 
            traindataloader, 
            valdataloader)