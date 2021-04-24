# -*- coding: utf-8 -*-
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model import BiLSTM_CRF_PARALLEL
from settings import EMBEDDING_DIM, HIDDEN_DIM, EPOCHS, TRAIN_DATA_PATH
from load_data import vocab2idx, label2idx, data_generator

device = "cuda" if torch.cuda.is_available() else 'cpu'

model = BiLSTM_CRF_PARALLEL(len(vocab2idx), label2idx, EMBEDDING_DIM, HIDDEN_DIM).to(device)
model.train()
#optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)
optimizer = optim.Adam(model.parameters(), lr=0.001)

start = time.time()

loss_vals = []
for epoch in range(EPOCHS):
    epoch_loss= []
    for sentence, tags in data_generator(TRAIN_DATA_PATH, vocab2idx, label2idx):
        sentences_idx_batch , tags_idx_batch  = sentence.to(device), tags.to(device)
        model.zero_grad()
        loss = model.neg_log_likelihood_parallel(sentences_idx_batch, tags_idx_batch)
        loss.backward()
        epoch_loss.append(loss.item())
        optimizer.step()
    loss_vals.append(np.mean(epoch_loss))    
    print(f'Epoch[{epoch}] - Loss:{np.mean(epoch_loss)}')

torch.save(model.state_dict(), "./saved_model/model.pth")
    
plt.plot(np.linspace(1, EPOCHS, EPOCHS).astype(int), loss_vals)
        
end = time.time()
print(f'Training costs:{end-start} seconds')