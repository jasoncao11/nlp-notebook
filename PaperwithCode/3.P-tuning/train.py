# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from model import PTuneForLAMA
from load_data import train_loader

device = "cuda" if torch.cuda.is_available() else 'cpu'
template = (3,3,3)
N_EPOCHS = 5
LR = 1e-5
MAX_GRAD_NORM = 1.0
DECAY_RATE = 0.98

model = PTuneForLAMA(device, template)
model.train()
optimizer = optim.Adam(model.parameters(), lr=LR)
my_lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=DECAY_RATE)

loss_vals_train = []
for epoch in range(N_EPOCHS):
    epoch_loss= []
    pbar = tqdm(train_loader)
    pbar.set_description("[Train Epoch {}]".format(epoch)) 

    for batch_idx, batch_data in enumerate(pbar):        
        model.zero_grad()
        loss = model(batch_data[0], batch_data[1])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        epoch_loss.append(loss.item())
        optimizer.step()
    my_lr_scheduler.step()
    loss_vals_train.append(np.mean(epoch_loss))
     
l1, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals_train)