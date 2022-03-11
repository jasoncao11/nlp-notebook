import torch
import torch.utils.data as tud
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

TRAIN_DATA_PATH = './data/train_data'
TEST_DATA_PATH = './data/test_data'
TOKENIZER_PATH = './bert-base-chinese'
BATCH_SIZE = 64
MAX_LEN = 512 #输入模型的最大长度，不能超过config中n_ctx的值
template = [("请找出句子中提及的机构","ORG"),("请找出句子中提及的地名","LOC"),("请找出句子中提及的人名","PER")]

def collate_fn(batch_data):
    """
    DataLoader所需的collate_fun函数，将数据处理成tensor形式
    Args:
        batch_data: batch数据
    Returns:
    """
    input_ids_list, token_type_ids_list, attention_mask_list, start_ids_list, end_ids_list = [], [], [], [], []
    for instance in batch_data:
        # 按照batch中的最大数据长度,对数据进行padding填充
        input_ids_temp = instance["input_ids"]
        token_type_ids_temp = instance["token_type_ids"]
        attention_mask_temp = instance["attention_mask"]
        start_ids_temp = instance["start_ids"]
        end_ids_temp = instance["end_ids"]
        # 添加到对应的list中
        input_ids_list.append(torch.tensor(input_ids_temp, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(token_type_ids_temp, dtype=torch.long))
        attention_mask_list.append(torch.tensor(attention_mask_temp, dtype=torch.long))
        start_ids_list.append(torch.tensor(start_ids_temp, dtype=torch.long))
        end_ids_list.append(torch.tensor(end_ids_temp, dtype=torch.long))
    # 使用pad_sequence函数，会将list中所有的tensor进行长度补全，补全到一个batch数据中的最大长度，补全元素为padding_value
    return {"input_ids": pad_sequence(input_ids_list, batch_first=True, padding_value=0),
            "token_type_ids": pad_sequence(token_type_ids_list, batch_first=True, padding_value=1),
            "attention_mask": pad_sequence(attention_mask_list, batch_first=True, padding_value=0),
            "start_ids": pad_sequence(start_ids_list, batch_first=True, padding_value=-100),
            "end_ids": pad_sequence(end_ids_list, batch_first=True, padding_value=-100)}
    
class NERDataset(tud.Dataset):
    def __init__(self, data_path, tokenizer_path, max_len):
        super(NERDataset, self).__init__()
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        chars = []
        labels = []
        self.data_set = []
        with open (data_path, encoding='utf8') as rf:
            for line in rf:
                if line != '\n':
                    char, label = line.strip().split()
                    chars.append(char)
                    if '-' in label:
                        label = label.split('-')[1]
                    labels.append(label)
                else:
                    for prefix, target in template:                       
                        input_ids_1 = [tokenizer.convert_tokens_to_ids(c) for c in prefix]
                        input_ids_1 = [tokenizer.cls_token_id] + input_ids_1 + [tokenizer.sep_token_id]
                        token_type_ids_1 = [0] * len(input_ids_1)
                        start_ids_1 = end_ids_1 = [-100] * len(input_ids_1)                       
                        if len(chars)+1+len(input_ids_1) > max_len:
                            chars = chars[:max_len-1-len(input_ids_1)]
                            labels = labels[:max_len-1-len(input_ids_1)]
                        input_ids_2 = [tokenizer.convert_tokens_to_ids(c) for c in chars]
                        input_ids_2 = input_ids_2 + [tokenizer.sep_token_id]
                        token_type_ids_2 = [1] * len(input_ids_2)
                        labels_ = labels + ['O']
                        start_ids_2, end_ids_2 = self.get_ids(target, labels_)
                        start_ids_2[-1] = -100
                        end_ids_2[-1] = -100                      
                        input_ids = input_ids_1 + input_ids_2
                        token_type_ids = token_type_ids_1 + token_type_ids_2
                        start_ids = start_ids_1 + start_ids_2
                        end_ids = end_ids_1 + end_ids_2
                        assert len(input_ids) == len(token_type_ids) == len(start_ids) == len(end_ids)
                        self.data_set.append({"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask":[1]*len(input_ids), "start_ids":start_ids, "end_ids":end_ids})
                    chars = []
                    labels = []
       
    @staticmethod                
    def get_ids(target, data):
        start_ids = [0]*len(data)
        end_ids = [0]*len(data)
        flag = 0
        for ind, t in enumerate(data):
            if not flag:
                if t == target:
                    start_ids[ind] = 1
                    flag = 1
            else:
                if t != target:
                    end_ids[ind-1] = 1
                    flag = 0
        if flag:
            end_ids[ind] = 1
        return(start_ids, end_ids)
               
    def __len__(self):
        return len(self.data_set)
    
    def __getitem__(self, idx):    
        return self.data_set[idx]
    
traindataset = NERDataset(TRAIN_DATA_PATH, TOKENIZER_PATH, MAX_LEN)
traindataloader = tud.DataLoader(traindataset, BATCH_SIZE, shuffle=True, collate_fn=collate_fn)