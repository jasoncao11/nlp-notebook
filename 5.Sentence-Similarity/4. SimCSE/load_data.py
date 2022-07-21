import csv
import torch
import torch.utils.data as tud
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

TRAIN_DATA_PATH = '../data/train_update_1.csv'
TOKENIZER_PATH = '../bert-base-chinese'
BATCH_SIZE = 128
MAX_LEN = 512

def collate_fn_train(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list_1, attention_mask_list_1 = [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp_1 = instance["input_ids_1"]
        attention_mask_temp_1 = instance["mask_1"]
        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list_1.append(torch.tensor(input_ids_temp_1, dtype=torch.long))
        attention_mask_list_1.append(torch.tensor(attention_mask_temp_1, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids_1": pad_sequence(input_ids_list_1, batch_first=True, padding_value=0),
            "attention_mask_1": pad_sequence(attention_mask_list_1, batch_first=True, padding_value=0)}

class SimiDataset_Train(tud.Dataset):
    def __init__(self, data_path, tokenizer_path, max_len):
        super(SimiDataset_Train, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        
        self.data_set = []
        with open (data_path, 'r', encoding='utf8') as rf:
            r = csv.reader(rf)
            for row in r:
                sent1 = row[0]
                tokens_1 = self.tokenizer.tokenize(sent1)
                if len(tokens_1) > self.max_len - 2:
                    tokens_1 = tokens_1[:self.max_len]
                input_ids_1 = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_1 + ['[SEP]'])
                mask_1 = [1] * len(input_ids_1)               
                for k in range(2):
                    self.data_set.append({"input_ids_1": input_ids_1, "mask_1": mask_1})
               
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        return self.data_set[idx]

traindataset = SimiDataset_Train(TRAIN_DATA_PATH, TOKENIZER_PATH, MAX_LEN)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn_train)

VAL_DATA_PATH = '../data/dev.csv'

def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list_1, attention_mask_list_1, input_ids_list_2, attention_mask_list_2, labels_list = [], [], [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp_1 = instance["input_ids_1"]
        attention_mask_temp_1 = instance["mask_1"]
        input_ids_temp_2 = instance["input_ids_2"]
        attention_mask_temp_2 = instance["mask_2"]
        label_temp = instance["label"]
        # 将input_ids_temp和token_type_ids_temp添加到对应的list中
        input_ids_list_1.append(torch.tensor(input_ids_temp_1, dtype=torch.long))
        attention_mask_list_1.append(torch.tensor(attention_mask_temp_1, dtype=torch.long))
        input_ids_list_2.append(torch.tensor(input_ids_temp_2, dtype=torch.long))
        attention_mask_list_2.append(torch.tensor(attention_mask_temp_2, dtype=torch.long))
        labels_list.append(label_temp)
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids_1": pad_sequence(input_ids_list_1, batch_first=True, padding_value=0),
            "attention_mask_1": pad_sequence(attention_mask_list_1, batch_first=True, padding_value=0),
            "input_ids_2": pad_sequence(input_ids_list_2, batch_first=True, padding_value=0),
            "attention_mask_2": pad_sequence(attention_mask_list_2, batch_first=True, padding_value=0),
            "labels": torch.tensor(labels_list, dtype=torch.long)}

class SimiDataset(tud.Dataset):
    def __init__(self, data_path, tokenizer_path, max_len):
        super(SimiDataset, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        
        self.data_set = []
        with open (data_path, 'r', encoding='utf8') as rf:
            r = csv.reader(rf)
            next(r)
            for row in r:
                sent1 = row[2]
                sent2 = row[3]
                label = int(row[4])
                
                tokens_1 = self.tokenizer.tokenize(sent1)
                if len(tokens_1) > self.max_len - 2:
                    tokens_1 = tokens_1[:self.max_len]
                input_ids_1 = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_1 + ['[SEP]'])
                mask_1 = [1] * len(input_ids_1)

                tokens_2 = self.tokenizer.tokenize(sent2)
                if len(tokens_2) > self.max_len - 2:
                    tokens_2 = tokens_2[:self.max_len]
                input_ids_2 = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens_2 + ['[SEP]'])
                mask_2 = [1] * len(input_ids_2)
                
                self.data_set.append({"input_ids_1": input_ids_1, "mask_1": mask_1, "input_ids_2": input_ids_2, "mask_2": mask_2, "label": label})
               
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):
        return self.data_set[idx]

valdataset = SimiDataset(VAL_DATA_PATH, TOKENIZER_PATH, MAX_LEN)
valdataloader = tud.DataLoader(valdataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)