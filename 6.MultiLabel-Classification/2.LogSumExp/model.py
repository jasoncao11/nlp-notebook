import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertPreTrainedModel

class TextRCNN_Bert(BertPreTrainedModel):
    def __init__(self, config):
        super(TextRCNN_Bert, self).__init__(config)
        self.bert = BertModel(config)
        self.lstm = nn.LSTM(768, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(128 * 2 + 768, 65)

    def forward(self, context, mask):
        outputs = self.bert(context, attention_mask=mask, output_hidden_states = True)
        #last_hidden_state = outputs[0] # [batch_size, seq_len, 768]
        hidden_states = outputs[2]
        second_to_last_layer = hidden_states[-2] # [batch_size, seq_len, 768]
        out, _ = self.lstm(second_to_last_layer) # [batch_size, seq_len, 128 * 2]
        out = torch.cat((second_to_last_layer, out), 2) # [batch_size, seq_len, 128 * 2 + 768]
        out = F.relu(out) # [batch_size, seq_len, 128 * 2 + 768]
        out = out.permute(0, 2, 1) # [batch_size, 128 * 2 + 768, seq_len]
        out = F.max_pool1d(out, out.size(2)).squeeze(2) # [batch_size, 128 * 2 + 768]
        logit = self.fc(out)
        return logit