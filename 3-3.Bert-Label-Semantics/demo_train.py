# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from model import LS_NER, Label_Encoder, Token_Encoder
from load_data import traindataloader
from load_label import label_input_ids, label_attention_mask

N_EPOCHS = 20
LR = 5e-4
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
MODEL_PATH = './bert-base-chinese'
device = "cuda" if torch.cuda.is_available() else 'cpu'

label_encoder = Label_Encoder.from_pretrained(MODEL_PATH)
token_encoder = Token_Encoder.from_pretrained(MODEL_PATH)

model = LS_NER(label_encoder, token_encoder)
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
    model.train()
    epoch_loss = []
    #num = 0
    pbar = tqdm(traindataloader)
    pbar.set_description("[Train Epoch {}]".format(epoch)) 
    for batch_idx, batch_data in enumerate(pbar):           
        token_input_ids = batch_data["input_ids"].to(device)
        token_attention_mask = batch_data["attention_mask"].to(device)
        label_ids = batch_data["label_ids"].to(device)
        #num += len(token_input_ids)
        model.zero_grad()
        loss = model(label_input_ids, label_attention_mask, token_input_ids, token_attention_mask, label_ids)           
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        epoch_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
        #if num >= 3000:
        #    break
    loss_vals.append(np.mean(epoch_loss))
plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals)

label_encoder.save_pretrained('./saved_model_label_encoder')
token_encoder.save_pretrained('./saved_model_token_encoder')