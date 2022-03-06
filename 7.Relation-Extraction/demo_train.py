import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from model import BertForRE
from load_data import traindataloader

N_EPOCHS = 10
LR = 5e-4
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
MODEL_PATH = './bert-base-chinese'
SAVED_DIR = './saved_model'
device = "cuda" if torch.cuda.is_available() else 'cpu'

model = BertForRE.from_pretrained(MODEL_PATH)
model.to(device)

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

total_steps = len(traindataloader) * N_EPOCHS
optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps), num_training_steps=total_steps)

loss_vals = []
for epoch in range(N_EPOCHS):
    num = 0
    model.train()
    epoch_loss = []
    pbar = tqdm(traindataloader)
    pbar.set_description("[Train Epoch {}]".format(epoch)) 
    for batch_idx, batch_data in enumerate(pbar):
        num += 1
        input_ids = batch_data['input_ids'].to(device)
        tag_ids = batch_data['tag_ids'].to(device)
        attention_mask = batch_data['attention_mask'].to(device)
        sub_mask = batch_data['sub_mask'].to(device)
        obj_mask = batch_data['obj_mask'].to(device)
        labels = batch_data['labels'].to(device)
        real_lens = batch_data['real_lens']
        model.zero_grad()
        loss = model.compute_loss(input_ids, attention_mask, tag_ids, sub_mask, obj_mask, labels, real_lens)           
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        epoch_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
        if num == 100:
            break
    loss_vals.append(np.mean(epoch_loss)) 

model.save_pretrained(f'{SAVED_DIR}')
plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals)