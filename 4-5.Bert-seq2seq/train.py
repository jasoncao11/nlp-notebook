# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from load_data import traindataloader, valdataloader
from tokenizer import load_chinese_base_vocab
from seq2seq_model import Seq2SeqModel

N_EPOCHS = 10
LR = 5e-4
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
MODEL_PATH = './bert-base-chinese/pytorch_model.bin'
VOCAB_PATH = './bert-base-chinese/vocab.txt'
word2idx, keep_tokens = load_chinese_base_vocab(VOCAB_PATH, simplfied=True)

def run():
    best_valid_loss = float('inf')
    model = Seq2SeqModel(word2idx)
    model.load_pretrain_params(MODEL_PATH, keep_tokens=keep_tokens)

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
        epoch_loss= []
        pbar = tqdm(traindataloader)
        pbar.set_description("[Train Epoch {}]".format(epoch)) 
    
        for batch_idx, batch_data in enumerate(pbar):
            
            input_ids = batch_data["input_ids"]
            token_type_ids = batch_data["token_type_ids"]
            token_type_ids_for_mask = batch_data["token_type_ids_for_mask"]
            target_ids = batch_data["target_ids"]
                       
            model.zero_grad()
            predictions, loss = model.forward(input_tensor=input_ids, 
                                              token_type_id=token_type_ids, 
                                              token_type_id_for_mask=token_type_ids_for_mask, 
                                              labels=target_ids)
            
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
                input_ids = batch_data["input_ids"]
                token_type_ids = batch_data["token_type_ids"]
                token_type_ids_for_mask = batch_data["token_type_ids_for_mask"]
                target_ids = batch_data["target_ids"]
                predictions, loss = model.forward(input_tensor=input_ids, 
                                                  token_type_id=token_type_ids, 
                                                  token_type_id_for_mask=token_type_ids_for_mask, 
                                                  labels=target_ids)                    
                epoch_loss_eval.append(loss.item())
                
        valid_loss = np.mean(epoch_loss_eval)
        loss_vals_eval.append(valid_loss)    
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model.save_all_params('model.pt')
        torch.cuda.empty_cache()
        
    l1, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals)
    l2, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals_eval)
    plt.legend(handles=[l1,l2],labels=['Train loss','Eval loss'],loc='best')

def predict(text):
    model = Seq2SeqModel(word2idx)
    model.eval()
    model.load_all_params('model.pt')
    print(model.sample_generate(text,top_k=5, top_p=0.95))
    
if __name__ == '__main__':
    run()