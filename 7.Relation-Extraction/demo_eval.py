import json
import torch
from transformers import BertTokenizer
from load_data import TOKENIZER_PATH, DEV_DATA_PATH, MAX_LEN
from model import BertForRE
from demo_train import SAVED_DIR
device = "cuda" if torch.cuda.is_available() else 'cpu'

model = BertForRE.from_pretrained(SAVED_DIR)
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
with open (DEV_DATA_PATH, encoding='utf8') as rf:
    for line in rf:
      data = json.loads(line)
      text = data["text"].lower()
      if len(text) <= MAX_LEN-2:
          print(text)
          spo_list = data["spo_list"]
          for spo in spo_list:
              print(f'{spo["subject"]} - {spo["predicate"]} - {spo["object"]}')
          chars = []
          for char in text:
              chars.append(char)
          chars = ['[CLS]'] + chars + ['[SEP]']                
          input_ids = [tokenizer.convert_tokens_to_ids(c) for c in chars]
          attention_mask = [1]*len(input_ids)
          input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
          attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)
          print(f'-----------------预测结果----------------')                                            
          model.predict(text, chars, input_ids, attention_mask)
          print(f'=========================================')