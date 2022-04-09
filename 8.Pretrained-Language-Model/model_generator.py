# coding=utf-8
import copy
import math
import torch
from torch import nn

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

class BertEmbeddings(nn.Module):
    """
        embeddings层
        构造word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.emb_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.emb_dim)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.emb_dim)
        self.LayerNorm = nn.LayerNorm(config.emb_dim, eps=config.layer_norm_eps)
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
        self.query = nn.Linear(config.emb_dim, self.all_head_size)
        self.key = nn.Linear(config.emb_dim, self.all_head_size)
        self.value = nn.Linear(config.emb_dim, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        #x: [batch size, seq len, hidden_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) #x: [batch size, seq len, num_attention_heads, attention_head_size]
        return x.permute(0, 2, 1, 3) #x: [batch size, num_attention_heads, seq len, attention_head_size]

    def forward(self, hidden_states, attention_mask):
        #hidden_states = [batch size, seq len, emb_dim]
            
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
        self.dense = nn.Linear(config.hidden_size, config.emb_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.emb_dim, eps=config.layer_norm_eps)
        
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
        self.dense = nn.Linear(config.emb_dim, config.intermediate_size)
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
        self.dense = nn.Linear(config.intermediate_size, config.emb_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(config.emb_dim, eps=config.layer_norm_eps)
        
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        #hidden_states = [batch size, seq len, emb_dim]
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

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.emb_dim, config.emb_dim)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.emb_dim, eps=config.layer_norm_eps)

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
        self.decoder = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

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

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.cls = BertOnlyMLMHead(config)
        
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=False):
        # input_ids： 一连串token在vocab中对应的id
        # token_type_id： 就是token对应的句子id,值为0或1（0表示对应的token属于第一句，1表示属于第二句）,当
        #                 任务只有一个句子输入时，token_type_id的每个值都是0，可不用传值
        # attention_mask：各元素的值为0或1,避免在padding的token上计算attention, 1不进行masked, 0则masked

        # 以上三个参数的shape为： (batch_size, sequence_length); type为tensor
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids).to(device)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids).to(device)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        #extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output, extended_attention_mask, output_all_encoded_layers=output_all_encoded_layers) 
        last_hidden_states = encoded_layers[-1]
        logits = self.cls(last_hidden_states)
        return logits