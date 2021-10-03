# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from load_data import traindataloader, valdataloader
from model import BertForSeq2Seq

N_EPOCHS = 30
LR = 5e-4
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
MODEL_PATH = './bert-base-chinese'
SAVE_PATH = './saved_models/pytorch_model.bin'
device = "cuda" if torch.cuda.is_available() else 'cpu'

def run():
    best_valid_loss = float('inf')
    model = BertForSeq2Seq.from_pretrained(MODEL_PATH)
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
    loss_vals_eval = []
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = []
        pbar = tqdm(traindataloader)
        pbar.set_description("[Train Epoch {}]".format(epoch)) 
    
        for batch_idx, batch_data in enumerate(pbar):
            
            input_ids = batch_data["input_ids"].to(device)
            token_type_ids = batch_data["token_type_ids"].to(device)
            token_type_ids_for_mask = batch_data["token_type_ids_for_mask"].to(device)
            labels = batch_data["labels"].to(device)
                       
            model.zero_grad()
            predictions, loss = model(input_ids, token_type_ids, token_type_ids_for_mask, labels)           
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            epoch_loss.append(loss.item())
            optimizer.step()
            scheduler.step()
        loss_vals.append(np.mean(epoch_loss))
        
        model.eval()
        epoch_loss_eval= []
        pbar = tqdm(valdataloader)
        pbar.set_description("[Eval Epoch {}]".format(epoch))
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(pbar):
                input_ids = batch_data["input_ids"].to(device)
                token_type_ids = batch_data["token_type_ids"].to(device)
                token_type_ids_for_mask = batch_data["token_type_ids_for_mask"].to(device)
                labels = batch_data["labels"].to(device)
                predictions, loss = model.forward(input_ids, token_type_ids, token_type_ids_for_mask, labels)                    
                epoch_loss_eval.append(loss.item())
                
        valid_loss = np.mean(epoch_loss_eval)
        loss_vals_eval.append(valid_loss)    
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), SAVE_PATH)
        torch.cuda.empty_cache()
   
    l1, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals)
    l2, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals_eval)
    plt.legend(handles=[l1,l2],labels=['Train loss','Eval loss'],loc='best')
    
if __name__ == '__main__':
    run()