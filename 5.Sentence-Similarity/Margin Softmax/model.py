import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertPreTrainedModel

device = "cuda" if torch.cuda.is_available() else 'cpu'

class Bert_Softmax(BertPreTrainedModel):
    def __init__(self, config):
        super(Bert_Softmax, self).__init__(config)
        self.bert = BertModel(config)
        self.fc = nn.Linear(768, 1509)

    def get_sents_embedding(self, context_1, mask_1, context_2, mask_2, method):
        outputs_1 = self.bert(context_1, attention_mask=mask_1, output_hidden_states=True)
        last_hidden_state_1 = outputs_1[0]# [batch_size, seq_len, 768]
        pooler_1 = outputs_1[1]# [batch_size, 768]

        outputs_2 = self.bert(context_2, attention_mask=mask_2, output_hidden_states=True)
        last_hidden_state_2 = outputs_2[0]# [batch_size, seq_len, 768]
        pooler_2 = outputs_2[1]# [batch_size, 768]

        if method == 'pooler':
            return pooler_1, pooler_2
        if method == 'max_pooling':
            out_1 = last_hidden_state_1.permute(0, 2, 1)
            out_1 = F.max_pool1d(out_1, out_1.size(2)).squeeze(2)
            out_2 = last_hidden_state_2.permute(0, 2, 1)
            out_2 = F.max_pool1d(out_2, out_2.size(2)).squeeze(2)
            return out_1, out_2
        if method == 'mean_pooling':
            out_1 = last_hidden_state_1.permute(0, 2, 1)
            out_1 = nn.AvgPool1d(out_1.size(2))(out_1).squeeze(2)
            out_2 = last_hidden_state_2.permute(0, 2, 1)
            out_2 = nn.AvgPool1d(out_2.size(2))(out_2).squeeze(2)
            return out_1, out_2

    def forward(self, context, mask, labels, method, scale=30, margin=0.35):
        outputs = self.bert(context, attention_mask=mask, output_hidden_states=True)
        last_hidden_state = outputs[0]# [batch_size, seq_len, 768]
        pooler = outputs[1]# [batch_size, 768]

        bs = pooler.size(0)        

        labels = labels.view(-1, 1)
        onehot = torch.zeros(bs, 1509).to(device)
        onehot.scatter_(dim=1, index=labels, value=1)

        if method == 'pooler':
            logits = self.fc(pooler)
            logits = onehot * (logits - margin) + (1 - onehot) * logits
            logits = logits*scale
            return logits

        if method == 'max_pooling':
            out = last_hidden_state.permute(0, 2, 1)
            out = F.max_pool1d(out, out.size(2)).squeeze(2)
            logits = self.fc(out)
            logits = onehot * (logits - margin) + (1 - onehot) * logits
            logits = logits*scale
            return logits

        if method == 'mean_pooling':
            out = last_hidden_state.permute(0, 2, 1)
            out = nn.AvgPool1d(out.size(2))(out).squeeze(2)
            logits = self.fc(out)
            logits = onehot * (logits - margin) + (1 - onehot) * logits
            logits = logits*scale
            return logits            
         
    def predict(self, context_1, mask_1, context_2, mask_2, threshold, method):
        embed_1, embed_2 = self.get_sents_embedding(context_1, mask_1, context_2, mask_2, method)
        simi = torch.cosine_similarity(embed_1, embed_2)
        pred = torch.where(simi>=threshold, 1, 0)
        pred = pred.cpu().numpy()
        return pred