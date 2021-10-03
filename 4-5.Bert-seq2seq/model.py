from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import json
import logging
import math
from io import open
import os
import torch
from torch import nn
from torch.nn import CrossEntropyLoss

logger = logging.getLogger(__name__)

CONFIG_NAME = 'config.json'
WEIGHTS_NAME = 'pytorch_model.bin'
device = "cuda" if torch.cuda.is_available() else 'cpu'

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

class BertConfig(object):
    """bert的参数配置
    """
    def __init__(self,
                 vocab_size_or_config_json_file=21128,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 pad_token_id=0,
                 layer_norm_eps=1e-12):
        """
        参数:
            vocab_size_or_config_json_file:
                可直接传入vocab的尺寸，来使用默认的配置，默认是中文vocab的词典大小；也可直接指定配置文件（json格式）的路径
            hidden_size:
                encoder层和pooler层的尺寸
            num_hidden_layers:
                Transformer架构中encoder的层数
            num_attention_heads:
                Transformer架构中encoder的每一个attention layer的attention heads的个数
            intermediate_size:
                Transformer架构中encoder的中间层的尺寸，也就是feed-forward的尺寸
            hidden_act:
                encoder和pooler层的非线性激活函数，目前支持gelu、swish
            hidden_dropout_prob: 
                在embeddings, encode和pooler层的所有全连接层的dropout概率
            attention_probs_dropout_prob:
                attention probabilities 的dropout概率
            max_position_embeddings:
                模型的最大序列长度
            type_vocab_size:
                token_type_ids的类型
            initializer_range:
                模型权重norm初始化的方差
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.pad_token_id = pad_token_id
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """从dict构造一个BertConfig实例"""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """从json文件中构造一个BertConfig实例，推荐使用"""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """序列化实例，并保存实例为json字符串"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """序列化实例，并保存实例到json文件"""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

class BertEmbeddings(nn.Module):
    """
        embeddings层
        构造word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        #构造position_ids，shape:[batch size, seq len]
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device).to(device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids).to(device)
        #构造token_type_ids，shape:[batch size, seq len]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids).to(device)
        #构造word, position and token_type embeddings    
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        #embeddings相加
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    """
        self attention层
        原理可看这篇博客: http://jalammar.github.io/illustrated-transformer/
    """
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        #x: [batch size, seq len, hidden_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) #x: [batch size, seq len, num_attention_heads, attention_head_size]
        return x.permute(0, 2, 1, 3) #x: [batch size, num_attention_heads, seq len, attention_head_size]

    def forward(self, hidden_states, attention_mask):
        #hidden_states = [batch size, seq len, hidden_size]
            
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        #mixed_query_layer = [batch size, seq len, hidden_size]
        #mixed_key_layer = [batch size, seq len, hidden_size]
        #mixed_value_layer = [batch size, seq len, hidden_size] 

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #query_layer = [batch size, num_attention_heads, seq len, attention_head_size]
        #key_layer = [batch size, num_attention_heads, seq len, attention_head_size]
        #value_layer = [batch size, num_attention_heads, seq len, attention_head_size]   
        
        # q和k执行点积, 获得attention score
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #attention_scores = [batch size, num_attention_heads, seq len, seq len]
        
        # 执行attention mask，对于padding部分的attention mask，
        # 值为-1000*(1-0)，经过softmax后，attention_probs几乎为0，所以不会attention到padding部分
        attention_scores = attention_scores + attention_mask

        # 将attention score 归一化到0-1
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        #context_layer = [batch size, num_attention_heads, seq len, attention_head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        #context_layer = [batch size, seq len, num_attention_heads, attention_head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        #context_layer = [batch size, seq len, hidden_size]
        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        # Add & Norm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    """
        实现 self attention + Add & Norm
    """
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        #hidden_states = [batch size, seq len, intermediate_size]
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        #hidden_states = [batch size, seq len, hidden_size]
        hidden_states = self.dropout(hidden_states)
        # Add & Norm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    """
        顺序为: Self Attention --> Add --> LayerNorm --> Feed Forward --> Add --> LayerNorm
        其中: Attention + Add + LayerNorm 构成了BertAttention
              Feed Forward的第一层linear 构成了BertIntermediate
              Feed Forward的第二层linear + Add + LayerNorm 构成了BertOutput
    """
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertEncoder(nn.Module):
    """
        多层Transformer, base版本12层, large版本24层
    """
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

class BertPooler(nn.Module):
    """
        得到pooler output, size = [batch size, hidden_size]
    """
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 这里取了最后一层的CLS位置的tensor作为pooler层的输入
        # 当然，理论上说，怎么取都行, 有些任务上, 取最后一层所有位置的平均值、最大值更好, 或者取倒数n层，再做concat等等, 这由你决定
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        #pooled_output = [batch size, hidden_size]
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertPredictionHeadTransform(nn.Module):
    """
        last hidden state 在经过 BertLMPredictionHead 处理前进行线性变换, size = [batch size, seq len, hidden_size]
    """
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
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

class BertLMPredictionHead(nn.Module):
    """
        得到 language model prediction head, 输出[batch size, seq len, vocab_size]
    """
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        #prediction_scores = [batch size, seq len, vocab_size]
        return prediction_scores

class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        #seq_relationship_score = [batch size, 2]
        return seq_relationship_score

class BertPreTrainingHeads(nn.Module):
    """
        MLM + NSP Heads
    """
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertPreTrainedModel(nn.Module):
    """ 
        加载预训练模型类, 只支持指定模型路径，所以必须先下载好需要的模型文件
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ 初始化权重
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # bert参数初始化, tf版本在linear和Embedding层使用的是截断正态分布, pytorch没有实现该函数, 
            # 此种初始化对于加载预训练模型后进行finetune没有任何影响，
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *inputs, **kwargs):
        """
            参数
                pretrained_model_path:
                    预训练模型权重以及配置文件的路径
                config:
                    BertConfig实例                   
        """
        config_file = os.path.join(pretrained_model_path, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        print("Load Model config from file: {}".format(config_file))
        weights_path = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        print("Load Model weights from file: {}".format(weights_path))
        # 实例化模型
        model = cls(config, *inputs, **kwargs)
        state_dict = torch.load(weights_path)
        # 加载state_dict到pytorch模型当中
        old_keys = []
        new_keys = []
        # 替换掉预训练模型的dict中的key与模型名称不匹配的问题
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)         
        # 更新state_dict的key
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        """
           下面的加载模型完全等价于module中的load_state_dict方法，但是由于每个key的前缀多了"bert."，这里就自行实现了类似load_state_dict方法的功能
        """

        prefix = '' if hasattr(model, 'bert') else 'bert'

        model_state_dict = model.state_dict()
        expected_keys = list(model_state_dict.keys())
        if prefix:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        loaded_keys = list(state_dict.keys())

        missing_keys = list(set(expected_keys) - set(loaded_keys))
        unexpected_keys = list(set(loaded_keys) - set(expected_keys))

        error_msgs = []
        # 复制state_dict, 为了_load_from_state_dict能修改它
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()

        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, [], [], error_msgs)
            for name, child in module._modules.items():
                if child is not None:                  
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')

        if len(missing_keys) > 0:
            print(f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at {weights_path} and are newly initialized: {missing_keys}")
        if len(unexpected_keys) > 0:
            print(f"Some weights of the model checkpoint at {weights_path} were not used when initializing {model.__class__.__name__}: {unexpected_keys}")
        if len(error_msgs) > 0:
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:{' '.join(error_msgs)}")
        return model

class BertModel(BertPreTrainedModel):
    """BERT 模型 ("Bidirectional Embedding Representations from a Transformer")
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False):
        # input_ids： 一连串token在vocab中对应的id
        # token_type_id： 就是token对应的句子id,值为0或1（0表示对应的token属于第一句，1表示属于第二句）
        # attention_mask：各元素的值为0或1,避免在padding的token上计算attention, 1不进行masked, 0则masked

        attention_mask = (1.0 - attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, attention_mask, output_all_encoded_layers=output_all_encoded_layers)
        # 如果需要返回所有隐藏层的输出，返回的encoded_layers包含了embedding_output，所以一共是13层
        encoded_layers.insert(0, embedding_output)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output

class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss() #Tokens with indices set to ``-100`` are ignored
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score

class BertForSeq2Seq(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSeq2Seq, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids, token_type_ids_for_mask, labels=None):

        seq_len = input_ids.shape[1]
        ## 构建特殊的mask       
        mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.float32).tril().to(device)        
        t1 = token_type_ids_for_mask.unsqueeze(1).unsqueeze(2).float().to(device)
        t2 = (token_type_ids_for_mask != -1).unsqueeze(1).unsqueeze(3).float().to(device)        
        attention_mask = ((mask+t1)*t2 > 0).float()

        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        
        prediction_scores = self.cls(sequence_output) #[batch size, seq len, vocab size]

        if labels is not None:
            ## 计算loss
            prediction_scores = prediction_scores[:, :-1].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss() #Tokens with indices set to ``-100`` are ignored
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            return prediction_scores, loss 
        else:
            return prediction_scores