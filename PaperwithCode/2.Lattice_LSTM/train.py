# -*- coding: utf-8 -*-
import time
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from model import LatticeLSTM
from load_data import char2idx, word2idx, label2idx, data_generator

character_size = len(char2idx)
word_size = len(word2idx)
embed_dim = 300
hidden_dim = 128

EPOCHS = 20
TRAIN_DATA_PATH = './data/train_data'
device = "cuda" if torch.cuda.is_available() else 'cpu'

model = LatticeLSTM(character_size, word_size, label2idx, embed_dim, hidden_dim).to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=0.001)

start = time.time()
loss_vals = []
for epoch in range(EPOCHS):
    epoch_loss= []
    #num = 0
    for sent, input_ids, input_words, labels_idx in data_generator(TRAIN_DATA_PATH, char2idx, word2idx, label2idx, shuffle=True):
        #num += 1        
        model.zero_grad()
        loss = model.neg_log_likelihood(input_ids, input_words, labels_idx)
        loss.backward()
        epoch_loss.append(loss.item())
        #print(f' num {num}, loss:{loss.item()}')
        optimizer.step()

        #if num == 3000:
        #    break

    loss_vals.append(np.mean(epoch_loss))    
    print(f'Epoch[{epoch}] - Loss:{np.mean(epoch_loss)}')

torch.save(model.state_dict(), "./saved_model/model_lattice.pth")    
plt.plot(np.linspace(1, EPOCHS, EPOCHS).astype(int), loss_vals)       
end = time.time()
print(f'Training costs:{end-start} seconds')