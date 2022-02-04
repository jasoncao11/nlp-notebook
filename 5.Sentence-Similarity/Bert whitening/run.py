import torch
import numpy as np
import torch.nn as nn
from sklearn import metrics
from transformers import BertModel
from load_data import traindataloader, valdataloader

BERT_PATH = '../bert-base-chinese'
device = "cuda" if torch.cuda.is_available() else 'cpu'
bert = BertModel.from_pretrained(BERT_PATH).to(device)
bert.eval()

def get_vecs():
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    bert = BertModel.from_pretrained('/content/gdrive/My Drive/bert-base-chinese').to(device)
    bert.eval()
    array = np.empty((0, 768))
    with torch.no_grad():
        for batch in traindataloader:
            input_ids, attention_mask = batch['input_ids_1'].to(device), batch['attention_mask_1'].to(device) 
            outputs = bert(input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden_state = outputs[0]# [batch_size, seq_len, 768]
            out = last_hidden_state.permute(0, 2, 1)
            out = nn.AvgPool1d(out.size(2))(out).squeeze(2)# [batch_size, 768]
            out = out.cpu().data.numpy()
            array = np.append(array, out, axis=0)
    return array

vecs = get_vecs()

def compute_kernel_bias(vecs):
    """计算kernel和bias
    vecs.shape = [num_samples, embedding_size]，
    最后的变换：y = (x + bias).dot(kernel)
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    return W, -mu

kernel, bias = compute_kernel_bias(vecs)
kernel = torch.from_numpy(kernel).to(device)
bias = torch.from_numpy(bias).to(device)

for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():        
        for batch in valdataloader:
            input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, labels = batch['input_ids_1'].to(device), batch['attention_mask_1'].to(device), batch['input_ids_2'].to(device), batch['attention_mask_2'].to(device), batch['labels'].to(device)
            
            outputs_1 = bert(input_ids_1, attention_mask=attention_mask_1, output_hidden_states=True)
            last_hidden_state_1 = outputs_1[0]# [batch_size, seq_len, 768]
            out_1 = last_hidden_state_1.permute(0, 2, 1)
            out_1 = nn.AvgPool1d(out_1.size(2))(out_1).squeeze(2)# [batch_size, 768]
            out_1 = torch.matmul((out_1 + bias), kernel)

            outputs_2 = bert(input_ids_2, attention_mask=attention_mask_2, output_hidden_states=True)
            last_hidden_state_2 = outputs_2[0]# [batch_size, seq_len, 768]
            out_2 = last_hidden_state_2.permute(0, 2, 1)# [batch_size, 768]
            out_2 = nn.AvgPool1d(out_2.size(2))(out_2).squeeze(2)
            out_2 = torch.matmul((out_2 + bias), kernel)
            
            simi = torch.cosine_similarity(out_1, out_2)
            pred = torch.where(simi>=t, 1, 0)
            pred = pred.cpu().numpy()

            predict_all = np.append(predict_all, pred)   
            truth = labels.cpu().numpy()
            labels_all = np.append(labels_all, truth)    
    acc = metrics.accuracy_score(labels_all, predict_all)
    print(f'Threshold-{t}: Accuracy on dev is {acc}')  