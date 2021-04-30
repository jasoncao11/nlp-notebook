# -*- coding: utf-8 -*-
import jieba
import torch
from load_data import UNK_IDX, SOS_IDX, EOS_IDX, vocab2id, id2vocab
from model import Encoder, Decoder, Seq2Seq

device = "cuda" if torch.cuda.is_available() else 'cpu' 
INPUT_DIM = len(id2vocab)
OUTPUT_DIM = len(id2vocab)
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

model = Seq2Seq(enc, dec, device).to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()

text = '为了更好地预测网络流量,提出了一种改进型Elman网络模型,并用文化算法对该模型进行了优化,获得了更佳的拟合度和预测性能。'
tokens = [tok for tok in jieba.cut(text)]
tokens_idx = [SOS_IDX] + [vocab2id.get(word, UNK_IDX) for word in tokens] + [EOS_IDX]
tokens_idx = torch.tensor(tokens_idx)
print(tokens_idx)
res = []
hidden, cell = model.encoder(tokens_idx.unsqueeze(0).to(device))
inputs = torch.tensor([SOS_IDX]).to(device)
for t in range(1, 35):
    output, hidden, cell = model.decoder(inputs, hidden, cell)
    inputs = output.argmax(1)
    word = id2vocab[inputs.item()]
    res.append(word)
    if word == '<EOS>':
        break
print(''.join(res))

