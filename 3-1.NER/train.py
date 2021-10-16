# -*- coding: utf-8 -*-
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model import BiLSTM_CRF
from load_data import char2idx, label2idx, data_generator

EMBEDDING_DIM = 300
HIDDEN_DIM = 64
BATCH_SIZE = 512
EPOCHS = 50
TRAIN_DATA_PATH = "./data/train_data" # 训练数据

device = "cuda" if torch.cuda.is_available() else 'cpu'
model = BiLSTM_CRF(len(char2idx), label2idx, EMBEDDING_DIM, HIDDEN_DIM).to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.001)

start = time.time()
loss_vals = []
for epoch in range(EPOCHS):
    epoch_loss= []
    for inputs_idx_batch, labels_idx_batch, real_lengths in data_generator(TRAIN_DATA_PATH, char2idx, label2idx, BATCH_SIZE):
        inputs_idx_batch, labels_idx_batch  = inputs_idx_batch.to(device), labels_idx_batch.to(device)
        model.zero_grad()
        loss = model.neg_log_likelihood(inputs_idx_batch, labels_idx_batch, real_lengths)
        loss.backward()
        epoch_loss.append(loss.item())
        optimizer.step()
    loss_vals.append(np.mean(epoch_loss))    
    print(f'Epoch[{epoch}] - Loss:{np.mean(epoch_loss)}')

torch.save(model.state_dict(), "./saved_model/model.pth")
plt.plot(np.linspace(1, EPOCHS, EPOCHS).astype(int), loss_vals)
end = time.time()
print(f'Training costs:{end-start} seconds')