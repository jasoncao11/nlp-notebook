# -*- coding: utf-8 -*-
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from load_data import char2id, intent2id, slot2id, traindataloader, valdataloader
from joint_model import Joint_model

DEVICE = "cuda" if torch.cuda.is_available() else 'cpu'
EMBEDDING_SIZE = 300
LSTM_HIDDEN_SIZE = 128
VOCAB_SIZE = len(char2id)
INTENT_NUM = len(intent2id)
SLOT_NUM = len(slot2id)
HEAD_NUM = 8
N_EPOCHS = 100
CLIP = 1
DROPOUT = 0.1

model = Joint_model(EMBEDDING_SIZE, LSTM_HIDDEN_SIZE, INTENT_NUM, SLOT_NUM, VOCAB_SIZE, HEAD_NUM, DROPOUT, DEVICE)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=5e-5)

slot_train_loss = []
intent_train_loss = []
slot_eval_loss = []
intent_eval_loss = []

for epoch in range(N_EPOCHS):
    model.train()
    epoch_slot_train_loss = []
    epoch_intent_train_loss = []
    pbar = tqdm(traindataloader)
    pbar.set_description("[Train Epoch {}]".format(epoch)) 

    for batch in pbar:
        input_ids = batch["input_ids"].to(DEVICE)
        slot_ids = batch["slot_ids"].to(DEVICE)
        intent_ids = batch["intent_ids"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        model.zero_grad()
        logits_intent, logits_slot = model.forward_logit(input_ids, mask)
        loss_intent, loss_slot = model.loss(logits_intent, logits_slot, intent_ids, slot_ids, mask)
        loss = loss_intent + loss_slot
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        epoch_slot_train_loss.append(loss_slot.item())
        epoch_intent_train_loss.append(loss_intent.item())
        optimizer.step()

    slot_train_loss.append(np.mean(epoch_slot_train_loss))
    intent_train_loss.append(np.mean(epoch_intent_train_loss))
    
    model.eval()
    epoch_slot_eval_loss = []
    epoch_intent_eval_loss = []
    pbar = tqdm(valdataloader)
    pbar.set_description("[Eval Epoch {}]".format(epoch))
    with torch.no_grad(): 
        for batch in pbar:
            input_ids = batch["input_ids"].to(DEVICE)
            slot_ids = batch["slot_ids"].to(DEVICE)
            intent_ids = batch["intent_ids"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            logits_intent, logits_slot = model.forward_logit(input_ids, mask)
            loss_intent, loss_slot = model.loss(logits_intent, logits_slot, intent_ids, slot_ids, mask)

            epoch_slot_eval_loss.append(loss_slot.item())
            epoch_intent_eval_loss.append(loss_intent.item())

    slot_eval_loss.append(np.mean(epoch_slot_eval_loss))
    intent_eval_loss.append(np.mean(epoch_intent_eval_loss))

# Save
torch.save(model.state_dict(), 'joint_model.pt')

plot1 = plt.figure(1)
l1, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), slot_train_loss)
l2, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), intent_train_loss)
plt.legend(handles=[l1,l2],labels=['Slot Train Loss','Intent Train Loss'],loc='best')

plot2 = plt.figure(2)
l3, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), slot_eval_loss)
l4, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), intent_eval_loss)
plt.legend(handles=[l3,l4],labels=['Slot Eval Loss','Intent Eval Loss'],loc='best')