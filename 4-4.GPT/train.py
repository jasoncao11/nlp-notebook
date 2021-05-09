# -*- coding: utf-8 -*-
import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers.modeling_gpt2 import GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup
from load_data import traindataloader, valdataloader

N_EPOCHS = 10
LR = 5e-4
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
SUMMARY_ID = 99
device = "cuda" if torch.cuda.is_available() else 'cpu'

def calculate_loss(outputs, labels, token_type_ids, summary_id):
    """
    只计算summary部分的loss
    """
    logits = outputs[0]  # 维度:[batch_size, sequence_length, config.vocab_size]   
    # 获取mask值，token_type_ids中等于summary_id的部分需要计算loss，标记为1；否则为0。
    # size:[batch_size, sequence_length]
    mask = (token_type_ids == summary_id).long()
    # 获取新的标签，size:[batch_size, sequence_length]
    labels = labels * mask    
    # 对预测结果和标签进行偏移操作
    # GPT2的生成机制为通过前面的token，预测下一个token；并且labels与input_ids相同，
    # 因此input_ids中的第一个token的预测结果，实际上是标签中的第二个token，以此类推，最终仅计算sequence_length-1个token的loss    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # 定义损失函数CrossEntropyLoss，并且设置忽略计算loss的索引，以及返回loss的形式
    # 忽略shift_labels中为0的loss，也就是仅计算title部分的损失值
    loss_fct = CrossEntropyLoss(ignore_index=0)
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return loss


def run():
    best_valid_loss = float('inf')
    model_config = GPT2Config.from_json_file('./config/config.json')
    model = GPT2LMHeadModel(config=model_config)
    model.to(device)
    
    total_steps = len(traindataloader) * N_EPOCHS
    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps), num_training_steps=total_steps)
    
    loss_vals = []
    loss_vals_eval = []
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss= []
        pbar = tqdm(traindataloader)
        pbar.set_description("[Train Epoch {}]".format(epoch)) 
    
        for batch_idx, batch_data in enumerate(pbar):
            input_ids = batch_data["input_ids"].to(device)
            token_type_ids = batch_data["token_type_ids"].to(device)
            model.zero_grad()
            outputs = model.forward(input_ids=input_ids)
            loss = calculate_loss(outputs, input_ids, token_type_ids, SUMMARY_ID)
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
                outputs = model.forward(input_ids=input_ids)
                loss = calculate_loss(outputs, input_ids, token_type_ids, SUMMARY_ID)       
                epoch_loss_eval.append(loss.item())
                
        valid_loss = np.mean(epoch_loss_eval)
        loss_vals_eval.append(valid_loss)    
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(f'model.pt')
        torch.cuda.empty_cache()
        
    l1, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals)
    l2, = plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals_eval)
    plt.legend(handles=[l1,l2],labels=['Train loss','Eval loss'],loc='best')
    
if __name__ == '__main__':
    run()

