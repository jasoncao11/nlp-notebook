import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel
from generalize_tensor import B_ORG, I_ORG, B_PER, I_PER, B_LOC, I_LOC


class Encoder(BertPreTrainedModel):
    def __init__(self, config):
        super(Encoder, self).__init__(config)
        self.bert = BertModel(config)
        for ind, t in enumerate([B_ORG, I_ORG, B_PER, I_PER, B_LOC, I_LOC], 1):
            self.bert.embeddings.word_embeddings.weight.data[ind] = t
        self.vs = config.vocab_size
        self.fc = nn.Linear(config.hidden_size, self.vs)
        self.gamma = 2
        self.alpha = 0.25
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        #self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, label_ids=None):
        x = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = x.last_hidden_state, x.pooler_output
        logits = self.fc(sequence_output)  # batch x seq_len x vocab_size

        if label_ids is not None:
            #loss = self.criterion(logits.view(-1, self.vs), label_ids.view(-1))
            loss = self.criterion(logits.view(-1, self.vs), label_ids.view(-1))
            pt = torch.exp(-loss)
            focal_loss = (self.alpha * (1-pt)**self.gamma * loss).mean()
            return focal_loss
        else:
            pred = torch.argmax(logits, dim=-1)
            return pred