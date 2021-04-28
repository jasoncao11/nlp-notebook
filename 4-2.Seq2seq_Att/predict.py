# -*- coding: utf-8 -*-
import jieba
import torch
from load_data import vocab_size, UNK_IDX, SOS_IDX, EOS_IDX, vocab2idx, idx2vocab
from model import Encoder, Decoder, Seq2Seq, Attention

device = "cuda" if torch.cuda.is_available() else 'cpu' 

INPUT_DIM = vocab_size
OUTPUT_DIM = vocab_size
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

attn = Attention(ENC_HID_DIM, DEC_HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)

model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()

text = '为了更好地预测网络流量,提出了一种改进型Elman网络模型,并用文化算法对该模型进行了优化,获得了更佳的拟合度和预测性能。'
tokens = [tok for tok in jieba.cut(text)]
tokens_idx = [SOS_IDX] + [vocab2idx.get(word, UNK_IDX) for word in tokens] + [EOS_IDX]
tokens_idx = torch.tensor(tokens_idx)
print(tokens_idx)
res = []
encoder_outputs, hidden = model.encoder(tokens_idx.unsqueeze(0).to(device))
inputs = torch.tensor([SOS_IDX]).to(device)
for t in range(1, 25):
    output, hidden = model.decoder(inputs, hidden, encoder_outputs)
    inputs = output.argmax(1)
    word = idx2vocab[inputs.item()]
    res.append(word)
    if word == '<EOS>':
        break
print(''.join(res))