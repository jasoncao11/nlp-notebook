import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from load_data import traindataloader, valdataloader
from model import Siamese_Net

SAVED_DIR = '../saved_model'
EPOCHS = 10
BERT_PATH = '../bert-base-chinese'
WARMUP_PROPORTION = 0.1
METHOD = 'mean_pooling'
device = "cuda" if torch.cuda.is_available() else 'cpu'

model = Siamese_Net.from_pretrained(BERT_PATH)
model.to(device)

total_steps = len(traindataloader) * EPOCHS
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps), num_training_steps=total_steps)

loss_vals = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss= []
    pbar = tqdm(traindataloader)
    pbar.set_description("[Epoch {}]".format(epoch)) 
    for batch in pbar:
        input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = batch['input_ids_1'].to(device), batch['attention_mask_1'].to(device), batch['input_ids_2'].to(device), batch['attention_mask_2'].to(device), batch['labels'].to(device)
        model.zero_grad()
        loss = model.compute_loss(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels, METHOD)
        loss.backward()
        epoch_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
    loss_vals.append(np.mean(epoch_loss)) 
    
    model.eval()
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():        
            for batch in valdataloader:
                input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = batch['input_ids_1'].to(device), batch['attention_mask_1'].to(device), batch['input_ids_2'].to(device), batch['attention_mask_2'].to(device), batch['labels'].to(device)
                pred = model.predict(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, t, METHOD)
                predict_all = np.append(predict_all, pred)   
                truth = labels.cpu().numpy()
                labels_all = np.append(labels_all, truth)    
        acc = metrics.accuracy_score(labels_all, predict_all)
        print(predict_all[:10])
        print(f'Epoch-{epoch} Threshold-{t}: Accuracy on dev is {acc}')       
    
model.save_pretrained(f'{SAVED_DIR}_{METHOD}')
plt.plot(np.linspace(1, EPOCHS, EPOCHS).astype(int), loss_vals)