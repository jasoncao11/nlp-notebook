import torch
import matplotlib.pyplot as plt
import numpy as np
from transformers import AdamW
from transformers import BertConfig, get_linear_schedule_with_warmup
from tqdm import tqdm
from generator import Generator
from discriminator import Discriminator
from electra import Electra
from load_data import data_generator

SAVED_DIR = './PLM_checkpoint'
EPOCHS = 100
WARMUP_PROPORTION = 0.1
device = "cuda" if torch.cuda.is_available() else 'cpu'

#generator
gen_config = BertConfig.from_json_file("./bert-base-chinese/config_gen.json")
generator = Generator(config=gen_config)
generator.embeddings.word_embeddings.weight = generator.cls.predictions.decoder.weight
#discriminator
discriminator = Discriminator.from_pretrained('./bert-base-chinese') 
#share the embeddings (both the token and positional embeddings) of the generator and discriminator
generator.embeddings = discriminator.bert.embeddings

electra = Electra(generator, discriminator, 1., 50.)
electra.to(device).train()
total_steps = EPOCHS
optimizer = AdamW(electra.parameters(), lr=5e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(WARMUP_PROPORTION * total_steps), num_training_steps=total_steps)

loss_vals = []
for epoch in range(EPOCHS):
    epoch_loss= []
    training_data = data_generator(12, 512, 10, './data/demo_data.csv', 2)
    pbar = tqdm(training_data)
    pbar.set_description("[Epoch {}]".format(epoch)) 
    for batch in pbar:
        original_input_ids = batch['original_input_ids'].to(device)
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
      
        electra.zero_grad()
        loss = electra(original_input_ids, input_ids, labels, attention_mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(electra.parameters(), 1.0)
        epoch_loss.append(loss.item())
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=loss.item())
    loss_vals.append(np.mean(epoch_loss))

discriminator.save_pretrained(f'{SAVED_DIR}')
plt.plot(np.linspace(1, EPOCHS, EPOCHS).astype(int), loss_vals)