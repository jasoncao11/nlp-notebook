import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from model import Encoder
from load_data import traindataloader

N_EPOCHS = 20
LR = 5e-4
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
MODEL_PATH = './bert-base-chinese'
SAVED_DIR = './saved_model'
device = "cuda" if torch.cuda.is_available() else 'cpu'


def run():
    model = Encoder.from_pretrained(MODEL_PATH)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    total_steps = len(traindataloader) * N_EPOCHS
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps),
                                                num_training_steps=total_steps)

    loss_vals = []
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = []
        pbar = tqdm(traindataloader)
        pbar.set_description("[Train Epoch {}]".format(epoch))
        for batch_idx, batch_data in enumerate(pbar):
            input_ids = batch_data["input_ids"].to(device)
            attention_mask = batch_data["attention_mask"].to(device)
            label_ids = batch_data["label_ids"].to(device)
            model.zero_grad()
            loss = model(input_ids, attention_mask, label_ids)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            epoch_loss.append(loss.item())
            optimizer.step()
            scheduler.step()
        loss_vals.append(np.mean(epoch_loss))
    print(loss_vals)
    model.save_pretrained(SAVED_DIR)
    #plt.plot(np.linspace(1, N_EPOCHS, N_EPOCHS).astype(int), loss_vals)


if __name__ == '__main__':
    run()