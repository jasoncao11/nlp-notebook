# -*- coding: utf-8 -*-
import jieba
import torch
from load_data import id2vocab, vocab2id, PAD_IDX, UNK_IDX, TEXT
from model import Encoder, Decoder, Transformer

device = "cuda" if torch.cuda.is_available() else 'cpu' 
INPUT_DIM = len(id2vocab)
OUTPUT_DIM = len(id2vocab)
HID_DIM = 512
ENC_LAYERS = 6
DEC_LAYERS = 6
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 2048
DEC_PF_DIM = 2048
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device)
dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device)
model = Transformer(enc, dec, PAD_IDX, device).to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()

sent = ''
tokens = [tok for tok in jieba.cut(sent)]
tokens = [TEXT.init_token] + tokens + [TEXT.eos_token]
    
src_indexes = [vocab2id.get(token, UNK_IDX) for token in tokens]
src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
src_mask = model.make_src_mask(src_tensor)

with torch.no_grad():
    enc_src = model.encoder(src_tensor, src_mask)

trg_indexes = [vocab2id[TEXT.init_token]]

for i in range(50):
    trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
    trg_mask = model.make_trg_mask(trg_tensor)  
    with torch.no_grad():
        output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
    
    pred_token = output.argmax(2)[:,-1].item()   
    trg_indexes.append(pred_token)

    if pred_token == vocab2id[TEXT.eos_token]:
        break

trg_tokens = [id2vocab[i] for i in trg_indexes]

print(trg_tokens[1:])