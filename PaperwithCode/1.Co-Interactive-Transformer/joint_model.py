# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from crf import CRF

class Label_Attention(nn.Module):
    def __init__(self, intent_emb, slot_emb):
        # intent_emb:[n_class, hidden_dim]
        # slot_emb:[n_tag, hidden_dim]
        super(Label_Attention, self).__init__()
        self.W_intent_emb = intent_emb.weight #[n_class, hidden_dim]
        self.W_slot_emb = slot_emb.weight #[n_tag, hidden_dim]

    def forward(self, input_intent, input_slot):
        # input_intent:[batch size, seq len, hidden_dim]
        # input_slot:[batch size, seq len, hidden_dim]
        intent_score = torch.matmul(input_intent, self.W_intent_emb.t()) #[batch size, seq len, n_class]
        slot_score = torch.matmul(input_slot, self.W_slot_emb.t()) #[batch size, seq len, n_tag]
        intent_probs = nn.Softmax(dim=-1)(intent_score) #[batch size, seq len, n_class]
        slot_probs = nn.Softmax(dim=-1)(slot_score) #[batch size, seq len, n_tag]
        intent_res = torch.matmul(intent_probs, self.W_intent_emb) #[batch size, seq len, hidden_dim]
        slot_res = torch.matmul(slot_probs, self.W_slot_emb) #[batch size, seq len, hidden_dim]
        return intent_res, slot_res

class I_S_Block(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout, device):
        super(I_S_Block, self).__init__()
        self.I_S_Attention = I_S_SelfAttention(hidden_size, n_heads, dropout, device)
        self.I_Out = SelfOutput(hidden_size, dropout)
        self.S_Out = SelfOutput(hidden_size, dropout)
        self.I_S_Feed_forward = Intermediate_I_S(hidden_size, dropout, device)

    def forward(self, H_intent_input, H_slot_input, mask):
        # H_intent_input: [batch size, seq len, hidden_dim]
        # H_slot_input: [batch size, seq len, hidden_dim]
        # mask: [batch size, seq len]
        H_intent, H_slot = self.I_S_Attention(H_intent_input, H_slot_input, mask)
        # H_intent: [batch size, seq len, hidden_dim]
        # H_slot: [batch size, seq len, hidden_dim]
        H_intent = self.I_Out(H_intent, H_intent_input) # [batch size, seq len, hidden_dim]
        H_slot = self.S_Out(H_slot, H_slot_input) # [batch size, seq len, hidden_dim]
        H_intent, H_slot = self.I_S_Feed_forward(H_intent, H_slot)
        # H_intent: [batch size, seq len, hidden_dim]
        # H_slot: [batch size, seq len, hidden_dim]
        return H_intent, H_slot

class Intermediate_I_S(nn.Module):
    def __init__(self, hidden_size, dropout, device):
        super(Intermediate_I_S, self).__init__()
        self.dense_in = nn.Linear(hidden_size * 6, hidden_size)
        self.intermediate_act_fn = nn.ReLU()
        self.dense_out = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm_I = nn.LayerNorm(hidden_size)
        self.LayerNorm_S = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, hidden_states_I, hidden_states_S):
        # hidden_states_I: [batch size, seq len, hidden_dim]
        # hidden_states_S: [batch size, seq len, hidden_dim]
        hidden_states_in = torch.cat([hidden_states_I, hidden_states_S], dim=2) # [batch size, seq len, hidden_dim*2]
        batch_size, seq_length, hidden = hidden_states_in.size()

        #context word window
        h_pad = torch.zeros(batch_size, 1, hidden).to(self.device) # [batch size, 1, hidden_dim*2]
        h_left = torch.cat([h_pad, hidden_states_in[:, :seq_length - 1, :]], dim=1) # [batch size, seq len, hidden_dim*2]
        h_right = torch.cat([hidden_states_in[:, 1:, :], h_pad], dim=1) # [batch size, seq len, hidden_dim*2]
        hidden_states_in = torch.cat([hidden_states_in, h_left, h_right], dim=2) # [batch size, seq len, hidden_dim*6]

        hidden_states = self.dense_in(hidden_states_in) # [batch size, seq len, hidden_dim]
        hidden_states = self.intermediate_act_fn(hidden_states) # [batch size, seq len, hidden_dim]
        hidden_states = self.dense_out(hidden_states) # [batch size, seq len, hidden_dim]
        hidden_states = self.dropout(hidden_states) # [batch size, seq len, hidden_dim]
        hidden_states_I_NEW = self.LayerNorm_I(hidden_states + hidden_states_I) # [batch size, seq len, hidden_dim]
        hidden_states_S_NEW = self.LayerNorm_S(hidden_states + hidden_states_S) # [batch size, seq len, hidden_dim]
        return hidden_states_I_NEW, hidden_states_S_NEW

class SelfOutput(nn.Module):
    def __init__(self, hidden_size, dropout):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.Layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        # hidden_states: [batch size, seq len, hidden_dim]
        # input_tensor: [batch size, seq len, hidden_dim]
        hidden_states = self.dense(hidden_states) # [batch size, seq len, hidden_dim]
        hidden_states = self.dropout(hidden_states) # [batch size, seq len, hidden_dim]
        hidden_states = self.Layer_norm(hidden_states + input_tensor) # [batch size, seq len, hidden_dim]
        return hidden_states

class I_S_SelfAttention(nn.Module):
    def __init__(self, hidden_size, n_heads, dropout, device):
        super(I_S_SelfAttention, self).__init__()

        assert hidden_size % n_heads == 0       
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.query_slot = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.key_slot = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.value_slot = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, intent, slot, mask):
        # intent: [batch size, seq len, hidden_dim]
        # slot: [batch size, seq len, hidden_dim]
        # mask: [batch size, seq len]
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2) #[batch size, 1, 1, seq len]
        attention_mask = (1.0 - extended_attention_mask) * -10000.0 #[batch size, 1, 1, seq len]
        batch_size = intent.shape[0] 

        mixed_query_layer = self.query(intent) # [batch size, seq len, hidden_dim]
        mixed_key_layer = self.key(slot) # [batch size, seq len, hidden_dim]
        mixed_value_layer = self.value(slot) # [batch size, seq len, hidden_dim]
        mixed_query_layer_slot = self.query_slot(slot) # [batch size, seq len, hidden_dim]
        mixed_key_layer_slot = self.key_slot(intent) # [batch size, seq len, hidden_dim]
        mixed_value_layer_slot = self.value_slot(intent) # [batch size, seq len, hidden_dim]

        query_layer = mixed_query_layer.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, seq len, head dim]
        key_layer = mixed_key_layer.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, seq len, head dim]
        value_layer = mixed_value_layer.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, seq len, head dim]  
        
        query_layer_slot = mixed_query_layer_slot.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, seq len, head dim]
        key_layer_slot = mixed_key_layer_slot.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, seq len, head dim]
        value_layer_slot = mixed_value_layer_slot.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3) # [batch size, n heads, seq len, head dim]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / self.scale # [batch size, n heads, seq len, seq len]
        attention_scores = attention_scores + attention_mask # [batch size, n heads, seq len, seq len]
        attention_probs = torch.softmax(attention_scores, dim = -1) # [batch size, n heads, seq len, seq len]               
        context_layer = torch.matmul(self.dropout(attention_probs), value_layer) # [batch size, n heads, seq len, head dim]  
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [batch size, seq len, n heads, head dim]              
        context_layer = context_layer.view(batch_size, -1, self.hidden_size) # [batch size, seq len, hidden_dim]      

        attention_scores_slot = torch.matmul(query_layer_slot, key_layer_slot.transpose(-1, -2)) / self.scale # [batch size, n heads, seq len, seq len]
        attention_scores_slot = attention_scores_slot + attention_mask # [batch size, n heads, seq len, seq len]
        attention_probs_slot = torch.softmax(attention_scores_slot, dim = -1) # [batch size, n heads, seq len, seq len]               
        context_layer_slot = torch.matmul(self.dropout(attention_probs_slot), value_layer_slot) # [batch size, n heads, seq len, head dim]  
        context_layer_slot = context_layer_slot.permute(0, 2, 1, 3).contiguous() # [batch size, seq len, n heads, head dim]              
        context_layer_slot = context_layer_slot.view(batch_size, -1, self.hidden_size) # [batch size, seq len, hidden_dim] 

        return context_layer, context_layer_slot

class Joint_model(nn.Module):
    def __init__(self, embed_dim, hidden_dim, n_class, n_tag, vocab_size, n_heads, dropout, device):
        super(Joint_model, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_class = n_class
        self.n_tag = n_tag
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.dropout = dropout
        self.device = device
        
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=0)
        self.biLSTM = nn.LSTM(self.embed_dim, self.hidden_dim // 2, bidirectional=True, batch_first=True)
        
        self.intent_fc = nn.Linear(self.hidden_dim, self.n_class)
        self.slot_fc = nn.Linear(self.hidden_dim, self.n_tag)
        self.I_S_Emb = Label_Attention(self.intent_fc, self.slot_fc)
        self.T_block = I_S_Block(self.hidden_dim, self.n_heads, self.dropout, self.device)
        self.crflayer = CRF(self.n_tag)
        self.criterion = nn.CrossEntropyLoss()

    def forward_logit(self, inputs, mask):
        # inputs:[batch size, seq len]
        # mask:[batch size, seq len]
        embeds = self.embed(inputs) # [batch size, seq len, embed_dim]
        H, (_, _) = self.biLSTM(embeds) #[batch size, seq len, hidden_dim]
        H_I, H_S = self.I_S_Emb(H, H)
        #H_I: [batch size, seq len, hidden_dim]
        #H_S: [batch size, seq len, hidden_dim]
        H_I, H_S = self.T_block(H_I + H, H_S + H, mask)
        #H_I: [batch size, seq len, hidden_dim]
        #H_S: [batch size, seq len, hidden_dim]      
        intent_input = F.max_pool1d((H_I + H).transpose(1, 2), H_I.size(1)).squeeze(2) #[batch size, hidden_dim]
        logits_intent = self.intent_fc(intent_input) #[batch size, n_class]
        logits_slot = self.slot_fc(H_S + H) #[batch size, seq len, n_tag]
        return logits_intent, logits_slot

    def loss(self, logits_intent, logits_slot, intent_label, slot_label, mask):
        loss_intent = self.criterion(logits_intent, intent_label)
        logits_slot = logits_slot.transpose(1, 0)
        slot_label = slot_label.transpose(1, 0)
        mask = mask.transpose(1, 0)
        loss_slot = -self.crflayer(logits_slot, slot_label, mask) / logits_intent.size()[0]
        return loss_intent, loss_slot

    def pred_intent_slot(self, logits_intent, logits_slot, mask):
        pred_intent = torch.max(logits_intent, 1)[1]
        mask = mask.transpose(1, 0)
        logits_slot = logits_slot.transpose(1, 0)
        pred_slot = self.crflayer.decode(logits_slot, mask=mask)
        return pred_intent, pred_slot