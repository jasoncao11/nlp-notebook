# -*- coding: utf-8 -*-
import torch
from torch import nn
from load_data import START_TAG, STOP_TAG, label2idx
from transformers import BertPreTrainedModel, BertModel

device = "cuda" if torch.cuda.is_available() else 'cpu'

def log_sum_exp(smat):
    '''
    for example:
    输入:
    tensor([[[0.5840, 0.6834, 0.8859, 0.6457],
         [0.3828, 0.6881, 0.3363, 0.3396],
         [0.9382, 0.5262, 0.4825, 0.4868]],
        [[0.3437, 0.0670, 0.6303, 0.8735],
         [0.2810, 0.3536, 0.8671, 0.1565],
         [0.4990, 0.4223, 0.2033, 0.6486]]])
    输出:
    tensor([[[1.7604, 1.7339, 1.6947, 1.5972]],
        [[1.4774, 1.3910, 1.7017, 1.7007]]])
    '''
    vmax = smat.max(dim=1, keepdim=True).values
    return (smat - vmax).exp().sum(axis=1, keepdim=True).log() + vmax

class BertForNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForNER, self).__init__(config)
        self.bert = BertModel(config)

        self.hidden_dim = config.hidden_size
        self.label2idx = label2idx
        self.label_size = len(label2idx)
        self.hidden2label = nn.Linear(self.hidden_dim, self.label_size)

        self.transitions = nn.Parameter(torch.randn(self.label_size, self.label_size)) #转移矩阵，表示从某一列的label转移至某一行的label的TransitionScore
        self.transitions.data[label2idx[START_TAG], :] = -10000
        self.transitions.data[:, label2idx[STOP_TAG]] = -10000

    def get_features(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = x.last_hidden_state, x.pooler_output
        feats = self.hidden2label(sequence_output)
        return feats

    def get_total_scores(self, frames, real_lengths):
        '''
        得到所有可能路径的分数总和
        '''
        #frames：[batch size, seq len, label_size]
        #real_lengths：[batch size]
        alpha = torch.full((frames.shape[0], self.label_size), -10000.0).to(device) #[batch size, label_size]
        alpha[:, self.label2idx[START_TAG]] = 0. #初始状态的EmissionScore. START_TAG是0, 其他都是很小的值 "-10000"
        alpha_ = torch.zeros((frames.shape[0], self.label_size)).to(device) #[batch size, label_size]
        frames = frames.transpose(0,1) #[seq len, batch size, label_size]
        index = 0 
        for frame in frames:
            index += 1
            #alpha.unsqueeze(-1):当前各状态的分值分布,[batch size, label_size, 1]
            #frame.unsqueeze(1):发射分值,[batch size, 1, label_size]
            #self.transitions.T:转移矩阵,[label_size, label_size]
            #三者相加会广播,维度为[batch size, label_size, label_size], log_sum_exp后的维度为[batch size, 1, label_size]
            alpha = log_sum_exp(alpha.unsqueeze(-1) + frame.unsqueeze(1) + self.transitions.T).squeeze(1)#[batch size, label_size]

            for idx, length in enumerate(real_lengths):
                if length == index:
                    alpha_[idx] = alpha[idx]
        #最后转到EOS，发射分值为0，转移分值为 self.transitions[[self.label2idx[STOP_TAG]], :].T
        #alpha.unsqueeze(-1): [batch size, label_size, 1]
        #self.transitions[[self.label2idx[STOP_TAG]], :].T: [label_size, 1]
        #三者相加会广播,维度为[batch size, label_size, 1], log_sum_exp后的维度为[batch size, 1, 1]
        alpha_ = log_sum_exp(alpha_.unsqueeze(-1) + 0 + self.transitions[[self.label2idx[STOP_TAG]], :].T).flatten()#[batch size]
        return alpha_    

    def get_golden_scores(self, frames, labels_idx_batch, real_lengths):
        '''
        得到正确路径的得分
        '''
        #frames[batch size, seq len, label_size]
        #labels_idx_batch:[batch size, seq len]
        #real_lengths：[batch size]

        score = torch.zeros(labels_idx_batch.shape[0]).to(device)#[batch size]
        score_ = torch.zeros(labels_idx_batch.shape[0]).to(device)#[batch size]
        labels = torch.cat([torch.full([labels_idx_batch.shape[0],1],self.label2idx[START_TAG], dtype=torch.long).to(device),labels_idx_batch], dim=1)#[batch size, seq len+1],注意不要+[STOP_TAG]; 结尾有处理
        index = 0
        for i in range(frames.shape[1]): # 沿途累加每一帧的转移和发射
            index += 1
            frame=frames[:,i,:]#[batch size, label_size]
            score += self.transitions[labels[:,i + 1], labels[:,i]] + frame[range(frame.shape[0]),labels[:,i + 1]]#[batch size]

            for idx, length in enumerate(real_lengths):
                if length == index:
                    score_[idx] = score[idx]

        score_ = score_ + self.transitions[self.label2idx[STOP_TAG], labels[:,-1]] #[batch size],加上到STOP_TAG的转移
        return score_

    def viterbi_decode(self, frames):
        backtrace = [] # 回溯路径;  backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是神马状态
        alpha = torch.full((1, self.label_size), -10000.).to(device)
        alpha[0][self.label2idx[START_TAG]] = 0
        for frame in frames:
            # 这里跟get_total_scores稍有不同: 需要求最优路径（而非一个总体分值）, 所以还要对smat求column_max
            smat = alpha.T + frame.unsqueeze(0) + self.transitions.T      
            val, idx = torch.max(smat, 0)
            backtrace.append(idx)
            alpha = val.unsqueeze(0)

        # 回溯路径
        smat = alpha.T + 0 + self.transitions[[self.label2idx[STOP_TAG]], :].T        
        val, idx = torch.max(smat, 0)
        best_tag_id = idx.item()
              
        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]): # 从[1:]开始，去掉开头的 START_TAG
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return val.item(), best_path[::-1] # 返回最优路径分值和最优路径

    def neg_log_likelihood(self, input_ids, attention_mask, labels_idx, real_lengths):
        feats = self.get_features(input_ids, attention_mask)
        total_scores = self.get_total_scores(feats, real_lengths)
        gold_score = self.get_golden_scores(feats, labels_idx, real_lengths)
        return torch.mean(total_scores - gold_score)

    def forward(self, input_ids, attention_mask):    
        feats = self.get_features(input_ids, attention_mask)
        feats = feats.squeeze(0) #[seq len, label_size]
        result = self.viterbi_decode(feats)
        return result