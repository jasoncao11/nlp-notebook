# -*- coding: utf-8 -*-
'''
saved_models文件夹包含两个文件：
1).在原有bert-base-chinese基础上fine-tune的pytorch_model.bin
2).配置文件config.json，和原有bert-base-chinese的配置文件一样
'''
import torch 
import torch.nn.functional as F
import numpy as np
from model import BertForSeq2Seq
from tokenizer import Tokenizer

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def sample_generate(text, model_path, out_max_length=40, top_k=30, top_p=0.0, max_length=512):
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    model = BertForSeq2Seq.from_pretrained(model_path)
    model.to(device)
    model.eval()

    input_max_length = max_length - out_max_length
    input_ids, token_type_ids, token_type_ids_for_mask, labels = Tokenizer.encode(text, max_length=input_max_length)

    input_ids = torch.tensor(input_ids, device=device, dtype=torch.long).view(1, -1)
    token_type_ids = torch.tensor(token_type_ids, device=device, dtype=torch.long).view(1, -1)
    token_type_ids_for_mask = torch.tensor(token_type_ids_for_mask, device=device, dtype=torch.long).view(1, -1)
    #print(input_ids, token_type_ids, token_type_ids_for_mask)
    output_ids = []

    with torch.no_grad(): 
        for step in range(out_max_length):
            scores = model(input_ids, token_type_ids, token_type_ids_for_mask)
            logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
            logit_score[Tokenizer.unk_id] = -float('Inf')
            
            # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
            for id_ in set(output_ids):
                logit_score[id_] /= 1.5                
            
            filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            if Tokenizer.sep_id == next_token.item():
                break
            output_ids.append(next_token.item())
            input_ids = torch.cat((input_ids, next_token.long().unsqueeze(0)), dim=1)
            token_type_ids = torch.cat([token_type_ids, torch.ones((1, 1), device=device, dtype=torch.long)], dim=1)
            token_type_ids_for_mask = torch.cat([token_type_ids_for_mask, torch.zeros((1, 1), device=device, dtype=torch.long)], dim=1)
            #print(input_ids, token_type_ids, token_type_ids_for_mask)

    return Tokenizer.decode(np.array(output_ids))

summary = sample_generate(text='针对仿人机器人上楼梯行走稳定性问题,进行了步态规划、稳定区域和稳定裕量的计算、控制策略等关键技术的研究。通过仿真,得到稳定行走姿态变化情况及各关节参数,为研究实时控制提供了依据。', model_path='./saved_models', top_k=5, top_p=0.95)
print(summary)