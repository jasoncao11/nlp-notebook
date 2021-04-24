# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from settings import START_TAG, STOP_TAG

device = "cuda" if torch.cuda.is_available() else 'cpu'

def log_sum_exp(smat):
    vmax = smat.max(dim=1, keepdim=True).values
    return (smat - vmax).exp().sum(axis=1, keepdim=True).log() + vmax

class BiLSTM_CRF_PARALLEL(nn.Module):

    def __init__(self, vocab_size, label2idx, embedding_dim, hidden_dim):
        super(BiLSTM_CRF_PARALLEL, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.label2idx = label2idx
        self.tagset_size = len(label2idx)

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[label2idx[START_TAG], :] = -10000
        self.transitions.data[:, label2idx[STOP_TAG]] = -10000
       
    def init_hidden(self, bs):
        return (torch.randn(2, bs, self.hidden_dim // 2).to(device),
                torch.randn(2, bs, self.hidden_dim // 2).to(device))

    def _forward_alg_parallel(self, frames):
        alpha = torch.full((frames.shape[0], self.tagset_size), -10000.0).to(device)
        alpha[:, self.label2idx[START_TAG]] = 0.
        for frame in frames.transpose(0,1):
            alpha = log_sum_exp(alpha.unsqueeze(-1) + frame.unsqueeze(1) + self.transitions.T).squeeze(1)
        alpha = log_sum_exp(alpha.unsqueeze(-1) + 0 + self.transitions[[self.label2idx[STOP_TAG]], :].T).flatten()
        return alpha

    def _get_lstm_features_parallel(self, sentences_idx_batch):
        hidden = self.init_hidden(len(sentences_idx_batch))
        embeds = self.word_embeds(sentences_idx_batch)
        lstm_out, hidden_out = self.lstm(embeds, hidden)    
        lstm_feats = self.hidden2tag(lstm_out)      
        return lstm_feats

    def _score_sentence_parallel(self, feats, tags_idx_batch):
        score = torch.zeros(tags_idx_batch.shape[0]).to(device)
        tags = torch.cat([torch.full([tags_idx_batch.shape[0],1],self.label2idx[START_TAG], dtype=torch.long).to(device),tags_idx_batch], dim=1).to(device)
        for i in range(feats.shape[1]):
            feat=feats[:,i,:]
            score = score + \
                    self.transitions[tags[:,i + 1], tags[:,i]] + feat[range(feat.shape[0]),tags[:,i + 1]]
        score = score + self.transitions[self.label2idx[STOP_TAG], tags[:,-1]]
        return score

    def _viterbi_decode_parallel(self, feats):
        backtrace = []
        alpha = torch.full((feats.shape[0], self.tagset_size), -10000.0).to(device)
        alpha[:, self.label2idx[START_TAG]] = 0.        
        for i in range(feats.shape[1]):
            feat=feats[:,i,:]
            smat = alpha.unsqueeze(-1) + feat.unsqueeze(1) + self.transitions.T
            alpha, idx = torch.max(smat, 1)
            backtrace.append(idx)
        smat = alpha.T + 0 + self.transitions[[self.label2idx[STOP_TAG]], :].T
        val, idx = torch.max(smat, 0)        
        best_path = [[x.item()] for x in idx]
        for bptrs_t in reversed(backtrace[1:]):
            for i in range(feats.shape[0]):
                best_tag_id = bptrs_t[i][best_path[i][-1]].item()
                best_path[i].append(best_tag_id)        
        result = []
        for score, tag in zip(val, best_path):
            result.append((score.item(), tag[::-1]))
        return result

    def neg_log_likelihood_parallel(self, sentences_idx_batch, tags_idx_batch):
        feats = self._get_lstm_features_parallel(sentences_idx_batch)
        forward_score = self._forward_alg_parallel(feats)
        gold_score = self._score_sentence_parallel(feats, tags_idx_batch)
        return torch.mean(forward_score - gold_score)

    def forward(self, sentences_idx_batch):      
        lstm_feats = self._get_lstm_features_parallel(sentences_idx_batch)
        result = self._viterbi_decode_parallel(lstm_feats)
        return result