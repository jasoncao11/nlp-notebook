import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

class Label_Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super(Label_Encoder, self).__init__(config)
        self.bert = BertModel(config)
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = x.last_hidden_state, x.pooler_output
        return pooled_output         

class Token_Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super(Token_Encoder, self).__init__(config)
        self.bert = BertModel(config)
    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = x.last_hidden_state, x.pooler_output
        return sequence_output
        
class LS_NER(nn.Module):
    def __init__(self, label_encoder, token_encoder):
        super(LS_NER, self).__init__()
        self.label_encoder = label_encoder
        self.token_encoder = token_encoder
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, label_input_ids, label_attention_mask, token_input_ids, token_attention_mask, label_ids=None):
        batch_size = token_input_ids.shape[0]
        
        label_logits = self.label_encoder(label_input_ids, label_attention_mask)
        label_logits = label_logits.transpose(1,0).repeat(batch_size,1,1) # batch size*768*7(number of labels)
        
        token_logits = self.token_encoder(token_input_ids, token_attention_mask)# batch size*seq*768
        
        logits = torch.matmul(token_logits, label_logits) # batch size*seq*7
        if label_ids is not None:
            loss = self.criterion(logits.view(-1, 7), label_ids.view(-1))
            return loss
        else:
            pred = torch.argmax(logits, dim=-1)
            return pred     