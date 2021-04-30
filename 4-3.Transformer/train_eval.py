# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from load_data import train_iter, val_iter, id2vocab, PAD_IDX
from model import Encoder, Decoder, Transformer

device = "cuda" if torch.cuda.is_available() else 'cpu' 
INPUT_DIM = len(id2vocab)
OUTPUT_DIM = len(id2vocab)
HID_DIM = 512
ENC_LAYERS = 6
DEC_LAYERS = 6
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 2048
DEC_PF_DIM = 2048
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1
N_EPOCHS = 10
CLIP = 1

enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)

model = Transformer(enc, dec, PAD_IDX, device).to(device)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
model.apply(initialize_weights)

optimizer = optim.Adam(model.parameters(), lr=5e-5)
#we ignore the loss whenever the target token is a padding token.
criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)

loss_vals = []
loss_vals_eval = []
for epoch in range(N_EPOCHS):
    model.train()
    epoch_loss= []
    pbar = tqdm(train_iter)
    pbar.set_description("[Train Epoch {}]".format(epoch)) 
    for batch in pbar:
        trg, src = batch.trg.to(device), batch.src.to(device)
        model.zero_grad()
        output, _ = model(src, trg[:,:-1])
        #trg = [batch size, trg len]
        #output = [batch size, trg len-1, output dim]        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)                     
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]     
        loss = criterion(output, trg)    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        epoch_loss.append(loss.item())
        optimizer.step()
        pbar.set_postfix(loss=loss.item())
    loss_vals.append(np.mean(epoch_loss))
    
    model.eval()
    epoch_loss_eval= []
    pbar = tqdm(val_iter)
    pbar.set_description("[Eval Epoch {}]".format(epoch)) 
    for batch in pbar:
        trg, src = batch.trg.to(device), batch.src.to(device)
        model.zero_grad()
        output, _ = model(src, trg[:,:-1])
        #trg = [batch size, trg len]
        #output = [batch size, trg len-1, output dim]        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)                     
        #trg = [(trg len - 1) * batch size]
        #output = [(trg len - 1) * batch size, output dim]   
        loss = criterion(output, trg)    
        epoch_loss_eval.append(loss.item())
        pbar.set_postfix(loss=loss.item())
    loss_vals_eval.append(np.mean(epoch_loss_eval))    
    
torch.save(model.state_dict(), 'model.pt')

l1, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals)
l2, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals_eval)
plt.legend(handles=[l1,l2],labels=['Train loss','Eval loss'],loc='best')