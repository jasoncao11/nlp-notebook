import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel

class MRCModel_CE_Focal(BertPreTrainedModel):
    def __init__(self, config):
        super(MRCModel_CE_Focal, self).__init__(config)
        self.bert = BertModel(config)
        self.start_fc = nn.Linear(config.hidden_size, 2)
        self.end_fc = nn.Linear(config.hidden_size, 2)
        self.gamma = 2
        self.alpha = 0.25
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, input_ids, attention_mask, token_type_ids, start_ids=None, end_ids=None):
        x = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output, pooled_output = x.last_hidden_state, x.pooler_output 
        start_logits = self.start_fc(sequence_output) # batch x seq_len x 2 
        end_logits = self.end_fc(sequence_output) # batch x seq_len x 2      
        
        if start_ids is not None and end_ids is not None:
            start_ce_loss = self.criterion(start_logits.view(-1, 2), start_ids.view(-1))
            start_pt = torch.exp(-start_ce_loss)
            start_focal_loss = (self.alpha * (1-start_pt)**self.gamma * start_ce_loss).mean()
            
            end_ce_loss = self.criterion(end_logits.view(-1, 2), end_ids.view(-1))
            end_pt = torch.exp(-end_ce_loss)
            end_focal_loss = (self.alpha * (1-end_pt)**self.gamma * end_ce_loss).mean()            
            
            return start_focal_loss+end_focal_loss
        else:
            start_pred = torch.argmax(start_logits, dim=-1)
            end_pred = torch.argmax(end_logits, dim=-1)
            return start_pred, end_pred