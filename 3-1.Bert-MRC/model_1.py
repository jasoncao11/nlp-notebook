import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel

class MRCModel_BCE(BertPreTrainedModel):
    def __init__(self, config):
        super(MRCModel_BCE, self).__init__(config)
        self.bert = BertModel(config)
        self.start_fc = nn.Linear(config.hidden_size, 1)
        self.end_fc = nn.Linear(config.hidden_size, 1)
        self.criterion = nn.BCELoss()
        
    def forward(self, input_ids, attention_mask, token_type_ids, start_ids=None, end_ids=None):
        x = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output, pooled_output = x.last_hidden_state, x.pooler_output 
        start_logits = self.start_fc(sequence_output) # batch x seq_len x 1 
        end_logits = self.end_fc(sequence_output) # batch x seq_len x 1
        if start_ids is not None and end_ids is not None:
            #start_loss
            start_prob = torch.sigmoid(start_logits)
            start_prob=torch.pow(start_prob,2)
            start_indices = torch.nonzero(start_ids != -100, as_tuple=True)
            start_prob = start_prob[start_indices].view(-1)
            start_ids = start_ids[start_indices].float()
            start_loss = self.criterion(start_prob, start_ids)
            #end_loss
            end_prob = torch.sigmoid(end_logits)
            end_prob=torch.pow(end_prob,2)
            end_indices = torch.nonzero(end_ids != -100, as_tuple=True)
            end_prob = end_prob[end_indices].view(-1)
            end_ids = end_ids[end_indices].float()
            end_loss = self.criterion(end_prob, end_ids)
            return start_loss + end_loss
        else:
            #start_pred
            start_pred = torch.sigmoid(start_logits)
            start_pred = start_pred.squeeze(-1)
            start_pred = torch.where(start_pred>0.5, 1, 0)
            #end_pred
            end_pred = torch.sigmoid(end_logits)
            end_pred = end_pred.squeeze(-1)
            end_pred = torch.where(end_pred>0.5, 1, 0)
            return start_pred, end_pred   