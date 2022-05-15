import torch
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

TOKENIZER_PATH = './bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)
device = "cuda" if torch.cuda.is_available() else 'cpu'

label_input_ids = []
label_attention_mask = []
label_semantics = ["其他","人物起始","人物内部","地点起始","地点内部","机构起始","机构内部"]
for ls in label_semantics:
    input_ids = [tokenizer.convert_tokens_to_ids(c) for c in ls]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    label_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
    attention_mask = [1]*len(input_ids)
    label_attention_mask.append(torch.tensor(attention_mask, dtype=torch.long))

label_input_ids = pad_sequence(label_input_ids, batch_first=True, padding_value=0)   
label_input_ids = label_input_ids.to(device)
label_attention_mask = pad_sequence(label_attention_mask, batch_first=True, padding_value=0)
label_attention_mask = label_attention_mask.to(device)