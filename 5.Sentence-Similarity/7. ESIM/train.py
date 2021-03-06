import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from load_data import traindataloader, valdataloader
from model import ESIM

EPOCHS = 30
BERT_PATH = 'bert-base-chinese'
WARMUP_PROPORTION = 0.1
METHOD = 'mean_pooling'
device = "cuda" if torch.cuda.is_available() else 'cpu'

model = ESIM(21128,300,300).to(device)

total_steps = len(traindataloader) * EPOCHS
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps), num_training_steps=total_steps)
criterion = nn.CrossEntropyLoss()

loss_vals = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss= []
    pbar = tqdm(traindataloader)
    pbar.set_description("[Epoch {}]".format(epoch)) 
    for batch in pbar:
        input_ids_1, attention_mask_1, len_1, input_ids_2, attention_mask_2, len_2, labels = batch['input_ids_1'].to(device), batch['attention_mask_1'].to(device), batch['len_1'].to(device), batch['input_ids_2'].to(device), batch['attention_mask_2'].to(device), batch['len_2'].to(device), batch['labels'].to(device)
        model.zero_grad()
        out = model(input_ids_1, attention_mask_1, len_1, input_ids_2, attention_mask_2, len_2)
        loss = criterion(out, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        epoch_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
    loss_vals.append(np.mean(epoch_loss))

    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():        
        for batch in valdataloader:
            input_ids_1, attention_mask_1, len_1, input_ids_2, attention_mask_2, len_2, labels = batch['input_ids_1'].to(device), batch['attention_mask_1'].to(device), batch['len_1'].to(device), batch['input_ids_2'].to(device), batch['attention_mask_2'].to(device), batch['len_2'].to(device), batch['labels'].to(device)
            out = model(input_ids_1, attention_mask_1, len_1, input_ids_2, attention_mask_2, len_2)
            pred = torch.argmax(out, dim=-1)
            predict_all = np.append(predict_all, pred.cpu().numpy())   
            truth = labels.cpu().numpy()
            labels_all = np.append(labels_all, truth)    
    acc = metrics.accuracy_score(labels_all, predict_all)
    print(f'Epoch-{epoch}: Accuracy on dev is {acc}')          
    
plt.plot(np.linspace(1, EPOCHS, EPOCHS).astype(int), loss_vals)