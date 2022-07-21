import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from load_data import traindataloader, valdataloader
from model import Bert_Simcse

SAVED_DIR = '../saved_model'
EPOCHS = 10
BERT_PATH = '../bert-base-chinese' #Dropout设置为0.5
WARMUP_PROPORTION = 0.1
METHOD = 'mean_pooling'
device = "cuda" if torch.cuda.is_available() else 'cpu'

model = Bert_Simcse.from_pretrained(BERT_PATH)
model.to(device)

total_steps = len(traindataloader) * EPOCHS
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps), num_training_steps=total_steps)

def simcse_unsup_loss(y_pred):
    """无监督的损失函数
    y_pred (tensor): bert的输出, [batch_size, 768]
    """
    # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
    y_true = torch.arange(y_pred.shape[0], device=device)
    y_true = (y_true - y_true % 2 * 2) + 1
    # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
    sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
    # 将相似度矩阵对角线置为很小的值, 消除自身的影响
    sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
    # 相似度矩阵除以温度系数
    sim = sim / 0.05
    # 计算相似度矩阵与y_true的交叉熵损失
    loss = F.cross_entropy(sim, y_true)
    return loss

loss_vals = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss= []
    pbar = tqdm(traindataloader)
    pbar.set_description("[Epoch {}]".format(epoch)) 
    for batch in pbar:
        input_ids_1, attention_mask_1 = batch['input_ids_1'].to(device), batch['attention_mask_1'].to(device)
        model.zero_grad()
        out = model(input_ids_1, attention_mask_1, METHOD)
        loss = simcse_unsup_loss(out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
        epoch_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
    loss_vals.append(np.mean(epoch_loss)) 

    model.eval()
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():        
            for batch in valdataloader:
                input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = batch['input_ids_1'].to(device), batch['attention_mask_1'].to(device), batch['input_ids_2'].to(device), batch['attention_mask_2'].to(device), batch['labels'].to(device)
                pred = model.predict(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, t, METHOD)
                predict_all = np.append(predict_all, pred)   
                truth = labels.cpu().numpy()
                labels_all = np.append(labels_all, truth)    
        acc = metrics.accuracy_score(labels_all, predict_all)
        print(f'Epoch-{epoch} Threshold-{t}: Accuracy on dev is {acc}')       
    
model.save_pretrained(f'{SAVED_DIR}_{METHOD}')
plt.plot(np.linspace(1, EPOCHS, EPOCHS).astype(int), loss_vals)