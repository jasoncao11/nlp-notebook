# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np 
from tokenizer import Tokenizer
from basic_bert import BasicBert
from bert_model import BertConfig, BertModel, BertLMPredictionHead

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

class Seq2SeqModel(BasicBert):
    def __init__(self, word2ix):
        super(Seq2SeqModel, self).__init__()
        self.word2ix = word2ix
        self.tokenizer = Tokenizer(word2ix)
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        config = BertConfig(len(word2ix))
        self.bert = BertModel(config)
        self.decoder = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)
            
        self.hidden_dim = config.hidden_size
        self.vocab_size = len(word2ix)

    def compute_loss(self, predictions, labels, target_mask):
        """
        target_mask : 句子a部分和pad部分全为0， 而句子b部分为1
        """
        predictions = predictions.view(-1, self.vocab_size)
        labels = labels.view(-1)
        target_mask = target_mask.view(-1).float()
        loss = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        return (loss(predictions, labels) * target_mask).sum() / target_mask.sum() ## 通过mask 取消 pad 和句子a部分预测的影响
    
    def forward(self, input_tensor, token_type_id, token_type_id_for_mask, labels=None):
        ## 传入输入，token type id ，还有句子a 和句子b的长度，注意都是传入一个batch数据
        ##  传入的几个值，在seq2seq 的batch iter 函数里面都可以返回
        input_tensor = input_tensor.to(self.device)
        token_type_id = token_type_id.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)
        input_shape = input_tensor.shape
        seq_len = input_shape[1]
        ## 构建特殊的mask       
        mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32, device=self.device).tril()        
        t1 = token_type_id_for_mask.unsqueeze(1).unsqueeze(2).float().to(self.device)
        t2 = (token_type_id_for_mask != -1).unsqueeze(1).unsqueeze(3).float().to(self.device)        
        mask = ((mask+t1)*t2 > 0).float()
       
        enc_layers, _ = self.bert(input_tensor, token_type_ids=token_type_id, attention_mask=mask)
        squence_out = enc_layers[-1] ## 取出来最后一层输出

        predictions = self.decoder(squence_out)

        if labels is not None:
            ## 计算loss
            ## 需要构建特殊的输出mask 才能计算正确的loss
            # 预测的值不用取最后sep符号的结果 因此是到-1
            predictions = predictions[:, :-1].contiguous()
            target_mask = token_type_id[:, 1:].contiguous()
            loss = self.compute_loss(predictions, labels, target_mask)
            return predictions, loss 
        else:
            return predictions

    def sample_generate(self, text, out_max_length=40, top_k=30, top_p=0.0, max_length=256):
        input_max_length = max_length - out_max_length
        token_ids, token_type_ids, token_type_ids_for_mask = self.tokenizer.encode(text, max_length=input_max_length)

        token_ids = torch.tensor(token_ids, device=self.device, dtype=torch.long).view(1, -1)
        token_type_ids = torch.tensor(token_type_ids, device=self.device, dtype=torch.long).view(1, -1)
        token_type_ids_for_mask = torch.tensor(token_type_ids_for_mask, device=self.device, dtype=torch.long).view(1, -1)
        #print(token_ids, token_type_ids, token_type_ids_for_mask)
        output_ids = []
        sep_id = self.word2ix["[SEP]"]
        with torch.no_grad(): 
            for step in range(out_max_length):
                scores = self.forward(token_ids, token_type_ids, token_type_ids_for_mask)
                logit_score = torch.log_softmax(scores[:, -1], dim=-1).squeeze(0)
                logit_score[self.word2ix["[UNK]"]] = -float('Inf')
                
                # 对于已生成的结果generated中的每个token添加一个重复惩罚项，降低其生成概率
                for id_ in set(output_ids):
                    logit_score[id_] /= 1.5                
                
                filtered_logits = top_k_top_p_filtering(logit_score, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                if sep_id == next_token.item():
                    break
                output_ids.append(next_token.item())
                token_ids = torch.cat((token_ids, next_token.long().unsqueeze(0)), dim=1)
                token_type_ids = torch.cat([token_type_ids, torch.ones((1, 1), device=self.device, dtype=torch.long)], dim=1)
                token_type_ids_for_mask = torch.cat([token_type_ids_for_mask, torch.zeros((1, 1), device=self.device, dtype=torch.long)], dim=1)
                #print(token_ids, token_type_ids, token_type_ids_for_mask)

        return self.tokenizer.decode(np.array(output_ids))

