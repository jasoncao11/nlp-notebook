import math
import torch
from transformers import BertModel, BertPreTrainedModel
from torch import nn

def gelu(x):
    """ gelu激活函数
        在GPT架构中，使用的是gelu函数的近似版本，公式如下:
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        参考：https://kexue.fm/archives/7309
        这里是直接求的解析解，就是原始论文给出的公式
        论文 https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    """swish激活函数
    """
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class PredictionHeadTransform(nn.Module):
    """
        last hidden state 在经过 PredictionHead 处理前进行线性变换, size = [batch size, seq len, hidden_size]
    """
    def __init__(self, config):
        super(PredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class PredictionHead(nn.Module):
    """
        输出[batch size, seq len]
    """
    def __init__(self, config):
        super(PredictionHead, self).__init__()
        self.transform = PredictionHeadTransform(config)
        self.dense = nn.Linear(config.hidden_size, 1)
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.dense(hidden_states).squeeze(-1)
        return hidden_states

class PreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(PreTrainingHeads, self).__init__()
        self.predictions = PredictionHead(config)
   
    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)  
        return prediction_scores

class Discriminator(BertPreTrainedModel):
    def __init__(self, config):
        super(Discriminator, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = PreTrainingHeads(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        last_hidden_states = outputs[0]
        pooler = outputs[1]# [batch_size, 768]
        logits = self.cls(last_hidden_states)
        return logits, pooler