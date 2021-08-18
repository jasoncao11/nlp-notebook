# -*- coding: utf-8 -*-
import csv
import torch
import torch.utils.data as tud
from torch.nn.utils.rnn import pad_sequence

TRAIN_DATA_PATH = './data/data_with_slots_intent_train.csv'
DEV_DATA_PATH = './data/data_with_slots_intent_dev.csv'
SLOT_PATH = './data/slot_maping.csv'
INTENT_PATH = './data/intent_maping.csv'
BATCH_SIZE = 64
MIN_FREQ = 1

#Make char dict
char2id = {'<pad>':0, '<unk>':1}
char2freq = {}
with open(TRAIN_DATA_PATH, 'r', encoding='utf8') as rf:
    r = csv.reader(rf)
    for row in r:
        data = row[0].split()[:-2]
        for each in data:
            char = each.split(':')[0]
            char2freq[char] = char2freq.get(char, 0) + 1
filtered_chars = [char for char, freq in char2freq.items() if freq >= MIN_FREQ]
for ind, char in enumerate(filtered_chars, 2):
    char2id[char] = ind

#Make slot dict
slot2id = {'<pad>':0}
with open(SLOT_PATH, 'r', encoding='utf8') as rf:
    r = csv.reader(rf)
    for ind, row in enumerate(r, 1):
        slot2id[row[1]] = ind
print(slot2id)
id2slot = {}
for k, v in slot2id.items():
    id2slot[v] = k

#Make intent dict
intent2id = {}
with open(INTENT_PATH, 'r', encoding='utf8') as rf:
    r = csv.reader(rf)
    for ind, row in enumerate(r, 0):
        intent2id[row[1]] = ind
id2intent = {}
for k, v in intent2id.items():
    id2intent[v] = k

def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, slot_ids_list, intent_id_list, mask_list = [], [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        slot_ids_temp = instance["slot_ids"]
        mask_temp = instance["mask"]
        # 将input_ids_temp和slot_ids_temp添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        slot_ids_list.append(torch.tensor(slot_ids_temp, dtype=torch.long))
        mask_list.append(torch.tensor(mask_temp, dtype=torch.long))
        intent_id_list.append(instance["intent_id"])
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "slot_ids": pad_sequence(slot_ids_list, batch_first=True, padding_value=0),
            "intent_ids": torch.tensor(intent_id_list, dtype=torch.long),
            "mask": pad_sequence(mask_list, batch_first=True, padding_value=0)}
    
class IntentDataset(tud.Dataset):
    def __init__(self, data_path):
        super(IntentDataset, self).__init__()  
        self.data_set = []
        with open (data_path, 'r', encoding='utf8') as rf:
            r = csv.reader(rf)
            for row in r:
                data = row[0].split()
                input_ids = []
                slot_ids = []
                intent_id = intent2id[data[-1]]
                mask = [1] * (len(data) - 2)
                for combo in data[:-2]:
                    char, slot = combo.split(':') 
                    input_ids.append(char2id.get(char, 1))
                    slot_ids.append(slot2id[slot])
                assert len(input_ids) == len(slot_ids)            
                self.data_set.append({"input_ids": input_ids, "slot_ids": slot_ids, "intent_id":intent_id, "mask":mask})
               
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):    
        return self.data_set[idx]
    
traindataset = IntentDataset(TRAIN_DATA_PATH)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

valdataset = IntentDataset(DEV_DATA_PATH)
valdataloader = tud.DataLoader(valdataset, BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
