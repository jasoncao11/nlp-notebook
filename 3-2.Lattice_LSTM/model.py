# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from load_data import START_TAG, STOP_TAG

device = "cuda" if torch.cuda.is_available() else 'cpu'

class CharLSTM(nn.Module):

    def __init__(self, character_size, embed_dim, hidden_dim):
        super(CharLSTM, self).__init__()
        self.character_size = character_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.char_embeds = nn.Embedding(self.character_size, self.embed_dim)

        self.wf = nn.Linear(self.hidden_dim+self.embed_dim, self.hidden_dim)
        self.wi = nn.Linear(self.hidden_dim+self.embed_dim, self.hidden_dim)
        self.wo = nn.Linear(self.hidden_dim+self.embed_dim, self.hidden_dim)
        self.wc = nn.Linear(self.hidden_dim+self.embed_dim, self.hidden_dim)
        
    def forward(self, x, h_pre, c_pre, words_cell_states=[]):
        "一个基本LSTM单元内部的计算(字符级别)，x代表字符所对应的ID。"
        #x:[1, 1]
        #h_pre:[1, hidden_dim]
        #c_pre:[1, hidden_dim]
        #words_cell_states: [[cell_state, weight],[],...]
        char_embedding = self.char_embeds(x) #[1, 1, embed_dim]
        char_embedding = char_embedding.squeeze(0) #[1, embed_dim]
        
        f = torch.sigmoid(self.wf(torch.cat([char_embedding, h_pre], dim=1))) #[1, hidden_dim]
        i = torch.sigmoid(self.wi(torch.cat([char_embedding, h_pre], dim=1))) #[1, hidden_dim]
        c_ = torch.tanh(self.wc(torch.cat([char_embedding, h_pre], dim=1))) #[1, hidden_dim]

        if not words_cell_states:
            c_cur = f*c_pre + i*c_ #[1, hidden_dim]

        else:
            cell_states = [c_]
            weights = [i]
            for cell_state, weight in words_cell_states:
                cell_states.append(cell_state)
                weights.append(weight)
            weights = torch.cat(weights, dim=0)
            weights = torch.softmax(weights, dim=0)
            cell_states = torch.cat(cell_states, dim=0)

            c_cur = torch.sum(weights*cell_states, dim=0).unsqueeze(0) #[1, hidden_dim]

        o = torch.sigmoid(self.wo(torch.cat([char_embedding, h_pre], dim=1))) #[1, hidden_dim]
        h_cur = o*torch.tanh(c_cur) #[1, hidden_dim]        
        return h_cur, c_cur
    
class WordLSTM(nn.Module):

    def __init__(self, word_size, embed_dim, hidden_dim):
        super(WordLSTM, self).__init__()
        self.word_size = word_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.word_embeds = nn.Embedding(self.word_size, self.embed_dim)

        self.w_f = nn.Linear(self.hidden_dim+self.embed_dim, self.hidden_dim)
        self.w_i = nn.Linear(self.hidden_dim+self.embed_dim, self.hidden_dim)
        self.w_c = nn.Linear(self.hidden_dim+self.embed_dim, self.hidden_dim)
        self.w_l = nn.Linear(self.hidden_dim+self.embed_dim, self.hidden_dim)

    def forward(self, x, h, c):
        '''
        h，c来自于单词首字对应的LSTM单元输出的Hidden State和Cell State，比如对于计算词组"长江大桥"时，h和c来自于"长"字；计算"大桥"时，h和c来自于"大"字。x代表单词所对应的ID。
        '''
        #x:[1, 1]
        #h:[1, hidden_dim]
        #c:[1, hidden_dim]

        word_embedding = self.word_embeds(x) #[1, 1, embed_dim]
        word_embedding = word_embedding.squeeze(0) #[1, embed_dim]
        
        i = torch.sigmoid(self.w_i(torch.cat([word_embedding, h], dim=1))) #[1, hidden_dim]
        f = torch.sigmoid(self.w_f(torch.cat([word_embedding, h], dim=1))) #[1, hidden_dim]
        c_ = torch.tanh(self.w_c(torch.cat([word_embedding, h], dim=1))) #[1, hidden_dim]      

        word_cell_state = f*c + i*c_ #[1, hidden_dim]
    
        return word_cell_state

    def get_weight(self, x_embed, c):
        '''
        额外的门控单元，根据当前字符和单词信息进行计算, x代表当前字符的字符向量，比如在计算"长江大桥"时，x为"桥"字的字向量, c为forward方法求出的'长江大桥'一词传递给"桥"字的word_cell_state
        '''
        #x_embed:[1, embed_dim]
        #c:[1, hidden_dim]

        word_weight = torch.sigmoid(self.w_l(torch.cat([x_embed, c], dim=1))) #[1, hidden_dim]
        return word_weight
    
def log_sum_exp(smat):
    """
    参数: smat 是 "status matrix", DP状态矩阵; 其中 smat[i][j] 表示 上一帧为i状态且当前帧为j状态的分值
    作用: 针对输入的【二维数组的每一列】, 各元素分别取exp之后求和再取log; 物理意义: 当前帧到达每个状态的分值(综合所有来源)
    例如: smat = [[ 1  3  9]
                 [ 2  9  1]
                 [ 3  4  7]]
         其中 smat[:, 2]= [9,1,7] 表示当前帧到达状态"2"有三种可能的来源, 分别来自上一帧的状态0,1,2
         这三条路径的分值求和按照log_sum_exp法则，展开 log_sum_exp(9,1,7) = log(exp(9) + exp(1) + exp(7)) = 3.964
         所以，综合考虑所有可能的来源路径，【当前帧到达状态"2"的总分值】为 3.964
         前两列类似处理，得到一个行向量作为结果返回 [ [?, ?, 3.964] ]
    注意数值稳定性技巧 e.g. 假设某一列中有个很大的数
    输入的一列 = [1, 999, 4]
    输出     = log(exp(1) + exp(999) + exp(4)) # 【直接计算会遭遇 exp(999) = INF 上溢问题】
            = log(exp(1-999)*exp(999) + exp(999-999)*exp(999) + exp(4-999)*exp(999)) # 每个元素先乘后除 exp(999)
            = log([exp(1-999) + exp(999-999) + exp(4-999)] * exp(999)) # 提取公因式 exp(999)
            = log([exp(1-999) + exp(999-999) + exp(4-999)]) + log(exp(999)) # log乘法拆解成加法
            = log([exp(1-999) + exp(999-999) + exp(4-999)]) + 999 # 此处exp(?)内部都是非正数，不会发生上溢
            = log([exp(smat[0]-vmax) + exp(smat[1]-vmax) + exp(smat[2]-vmax)]) + vmax # 符号化表达
    代码只有两行，但是涉及二维张量的变形有点晦涩，逐步分析如下, 例如:
    smat = [[ 1  3  9]
            [ 2  9  1]
            [ 3  4  7]]
    smat.max(dim=0, keepdim=True) 是指【找到各列的max】，即: vmax = [[ 3  9  9]] 是个行向量
    然后 smat-vmax, 两者形状分别是 (3,3) 和 (1,3), 相减会广播(vmax广播复制为3*3矩阵)，得到:
    smat - vmax = [[ -2  -6  0 ]
                   [ -1  0   -8]
                   [ 0   -5  -2]]
    然后.exp()是逐元素求指数
    然后.sum(axis=0, keepdim=True) 是"sum over axis 0"，即【逐列求和】, 得到的是行向量，shape=(1,3)
    然后.log()是逐元素求对数
    最后再加上 vmax; 两个行向量相加, 结果还是个行向量
    """
    vmax = smat.max(dim=0, keepdim=True).values  # 每一列的最大数
    return (smat - vmax).exp().sum(axis=0, keepdim=True).log() + vmax

class LatticeLSTM(nn.Module):

    def __init__(self, character_size, word_size, label2idx, embed_dim, hidden_dim):
        super(LatticeLSTM, self).__init__()

        self.label2idx = label2idx
        self.hidden_dim = hidden_dim
        self.label_size = len(label2idx)
        self.hidden2label = nn.Linear(hidden_dim, self.label_size)
        self.charlstm = CharLSTM(character_size, embed_dim, hidden_dim)
        self.wordlstm = WordLSTM(word_size, embed_dim, hidden_dim)

        self.transitions = nn.Parameter(torch.randn(self.label_size, self.label_size)) #转移矩阵，表示从某一列的label转移至某一行的label的TransitionScore
        self.transitions.data[label2idx[START_TAG], :] = -10000
        self.transitions.data[:, label2idx[STOP_TAG]] = -10000

    def init_hidden(self):
        return (torch.randn(1, self.hidden_dim).to(device), torch.randn(1, self.hidden_dim).to(device))

    def get_lstm_features(self, input_ids, input_words):
        '''
        text = '水帘洞'
        input_ids = [0,2,1] #[char_id_1,...]
        input_words = [[[0,2],[1,3]],[],[]] #[[[word_id_1, word_length_1],[word_id_2, word_length_2]],[],...]
        '''
        char_h, char_c = self.init_hidden()

        length = len(input_ids)
        words_cell_states = [[]]*length

        hidden_states = []
        for idx, charid in enumerate(input_ids):
          charid = torch.tensor([[charid]]).to(device)
          char_h, char_c = self.charlstm(charid, char_h, char_c, words_cell_states[idx])
          hidden_states.append(char_h)
          if input_words[idx]:
            for word_id, word_length in input_words[idx]:
              word_id = torch.tensor([[word_id]]).to(device)
              word_cell_state = self.wordlstm(word_id, char_h, char_c)

              end_char_id = input_ids[idx+word_length-1]
              end_char_id = torch.tensor([[end_char_id]]).to(device)
              end_char_embed = self.charlstm.char_embeds(end_char_id).squeeze(0)
              word_weight = self.wordlstm.get_weight(end_char_embed, word_cell_state)

              words_cell_states[idx+word_length-1].append([word_cell_state, word_weight])
      
        hidden_states = torch.cat(hidden_states, dim=0) #[seq_len, hidden_dim]
        lstm_feats = self.hidden2label(hidden_states) #[seq_len, label_size]
        return lstm_feats

    def get_golden_score(self, lstm_feats, labels_idx):
        labels_idx.insert(0, self.label2idx[START_TAG])
        labels_tensor = torch.tensor(labels_idx).to(device)  # 注意不要+[STOP_TAG]; 结尾有处理
        score = torch.zeros(1).to(device)
        for i, frame in enumerate(lstm_feats):  # 沿途累加每一帧的转移和发射
            score += self.transitions[labels_tensor[i + 1], labels_tensor[i]] + frame[labels_tensor[i + 1]]
        return score + self.transitions[self.label2idx[STOP_TAG], labels_tensor[-1]]  # 加上到STOP_TAG的转移  

    def get_total_score(self, lstm_feats):
        """ 给定每一帧的发射分值; 按照当前的CRF层参数算出所有可能序列的分值和，用作概率归一化分母 """
        alpha = torch.full((1, self.label_size), -10000.0).to(device)
        alpha[0][self.label2idx[START_TAG]] = 0  # 初始化分值分布. START_TAG是log(1)=0, 其他都是很小的值 "-10000"
        for frame in lstm_feats:
            # log_sum_exp()内三者相加会广播: 当前各状态的分值分布(列向量) + 发射分值(行向量) + 转移矩阵(方形矩阵)
            # 相加所得矩阵的物理意义见log_sum_exp()函数的注释; 然后按列求log_sum_exp得到行向量
            alpha = log_sum_exp(alpha.T + frame.unsqueeze(0) + self.transitions.T)
        # 最后转到EOS，发射分值为0，转移分值为列向量 self.transitions[:, [self.label2idx[STOP_TAG]]]
        return log_sum_exp(alpha.T + 0 + self.transitions[[self.label2idx[STOP_TAG]], :].T).flatten()

    def viterbi_decode(self, lstm_feats):
        backtrace = [] # 回溯路径;  backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是神马状态
        alpha = torch.full((1, self.label_size), -10000.).to(device)
        alpha[0][self.label2idx[START_TAG]] = 0
        for frame in lstm_feats:
            # 这里跟get_total_scores稍有不同: 需要求最优路径（而非一个总体分值）, 所以还要对smat求column_max
            smat = alpha.T + frame.unsqueeze(0) + self.transitions.T
            
            val, idx = torch.max(smat, 0)
            backtrace.append(idx)
            alpha = val.unsqueeze(0)
        # 回溯路径
        smat = alpha.T + 0 + self.transitions[[self.label2idx[STOP_TAG]], :].T
        
        val, idx = torch.max(smat, 0)
        best_tag_id = idx.item()
              
        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]): # 从[1:]开始，去掉开头的 START_TAG
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return val.item(), best_path[::-1] # 返回最优路径分值和最优路径

    def neg_log_likelihood(self, input_ids, input_words, labels_idx):
        lstm_feats = self.get_lstm_features(input_ids, input_words)
        #print(lstm_feats)
        total_score = self.get_total_score(lstm_feats)
        gold_score = self.get_golden_score(lstm_feats, labels_idx)
        return total_score - gold_score

    def forward(self, input_ids, input_words):   
        lstm_feats = self.get_lstm_features(input_ids, input_words)
        result = self.viterbi_decode(lstm_feats)
        return result