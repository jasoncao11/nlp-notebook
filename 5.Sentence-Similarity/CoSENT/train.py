import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from load_data import traindataloader, valdataloader
from model import Bert_CoSENT

SAVED_DIR = '../saved_model'
EPOCHS = 10
BERT_PATH = '../bert-base-chinese'
WARMUP_PROPORTION = 0.1
METHOD = 'mean_pooling'
device = "cuda" if torch.cuda.is_available() else 'cpu'

model = Bert_CoSENT.from_pretrained(BERT_PATH)
model.to(device)

total_steps = len(traindataloader) * EPOCHS
optimizer = AdamW(model.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps), num_training_steps=total_steps)

def cal_loss(simi_scores, labels):
    #simi_scores, labels 都为1维tensor; simi_scores是每个句子对的余弦相似度，labels对应改句子对的标签
    neg_indices = torch.nonzero(labels != 1, as_tuple=True)
    pos_indices = torch.nonzero(labels != 0, as_tuple=True)
    neg = simi_scores[neg_indices]
    pos = simi_scores[pos_indices]
    neg = neg[:, None]
    pos = pos[None, :]
    #取出负例-正例的差值
    diff = neg-pos
    diff = diff.view(1,-1)   
    diff = torch.cat((torch.tensor([[0]]).float().to(device), diff), dim=1)
    return torch.logsumexp(diff, 1)

loss_vals = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss= []
    pbar = tqdm(traindataloader)
    pbar.set_description("[Epoch {}]".format(epoch)) 
    for batch in pbar:
        input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = batch['input_ids_1'].to(device), batch['attention_mask_1'].to(device), batch['input_ids_2'].to(device), batch['attention_mask_2'].to(device), batch['labels'].to(device)
        model.zero_grad()
        simi_scores = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, METHOD)
        loss = cal_loss(simi_scores, labels)
        loss.backward()
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