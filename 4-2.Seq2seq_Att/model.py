# -*- coding: utf-8 -*-
import random
import torch.nn as nn
import torch 
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()       
        self.embedding = nn.Embedding(input_dim, emb_dim)       
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, batch_first=True)     
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)     
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):     
        #src = [batch size, src len]
        embedded = self.dropout(self.embedding(src))
        #embedded = [batch size, src len, emb dim]
        outputs, hidden = self.rnn(embedded)
        #outputs = [batch size, src len, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]
        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        #hidden [-2, :, : ] is the last of the forwards RNN 
        #hidden [-1, :, : ] is the last of the backwards RNN
        #initial decoder hidden is final hidden state of the forwards and backwards 
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))
        #outputs = [batch size, src len, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        return outputs, hidden
    
class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()       
        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)
        
    def forward(self, hidden, encoder_outputs):        
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]
        src_len = encoder_outputs.shape[1]     
        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)                
        #hidden = [batch size, src len, dec hid dim]      
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2))) 
        #energy = [batch size, src len, dec hid dim]
        attention = self.v(energy).squeeze(2)        
        #attention= [batch size, src len]        
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention       
        self.embedding = nn.Embedding(output_dim, emb_dim)        
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, batch_first=True)        
        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)      
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs, hidden, encoder_outputs):             
        #inputs = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]        
        inputs = inputs.unsqueeze(1)
        #inputs = [batch size, 1]        
        embedded = self.dropout(self.embedding(inputs))
        #embedded = [batch size, 1, emb dim]        
        a = self.attention(hidden, encoder_outputs)                
        #a = [batch size, src len]     
        a = a.unsqueeze(1)        
        #a = [batch size, 1, src len]
        weighted = torch.bmm(a, encoder_outputs)      
        #weighted = [batch size, 1, enc hid dim * 2]     
        rnn_input = torch.cat((embedded, weighted), dim = 2)    
        #rnn_input = [batch size, 1, (enc hid dim * 2) + emb dim]           
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))       
        #output = [batch size, seq len, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]    
        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [batch size, 1, dec hid dim]
        #hidden = [1, batch size, dec hid dim]       
        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        weighted = weighted.squeeze(1)        
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))        
        #prediction = [batch size, output dim]        
        return prediction, hidden.squeeze(0)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()      
        self.encoder = encoder
        self.decoder = decoder
        self.device = device        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        #src = [batch size, src len]
        #trg = [batch size, trg len]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim   
        #tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)
                
        #first inputs to the decoder is the <sos> tokens
        inputs = trg[:,0]
        
        for t in range(1, trg_len):
            
            #insert inputs token embedding, previous hidden state and all encoder hidden states
            #receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(inputs, hidden, encoder_outputs)
            
            #place predictions in a tensor holding predictions for each token
            outputs[:,t,:] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next inputs
            #if not, use predicted token
            inputs = trg[:,t] if teacher_force else top1

        return outputs    