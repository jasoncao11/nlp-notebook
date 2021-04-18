# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)  # 可以复现官网的结果 https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
START_TAG, END_TAG = "<s>", "<e>"

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

class BiLSTM_CRF(nn.Module):
    def __init__(self, tag2ix, word2ix, embedding_dim, hidden_dim):
        """
        :param tag2ix: 序列标注问题的 标签 -> 下标 的映射
        :param word2ix: 输入单词 -> 下标 的映射
        :param embedding_dim: 喂进BiLSTM的词向量的维度
        :param hidden_dim: 期望的BiLSTM输出层维度
        """
        super(BiLSTM_CRF, self).__init__()
        assert hidden_dim % 2 == 0, 'hidden_dim must be even for Bi-Directional LSTM'
        self.embedding_dim, self.hidden_dim = embedding_dim, hidden_dim
        self.tag2ix, self.word2ix, self.n_tags = tag2ix, word2ix, len(tag2ix)

        self.word_embeds = nn.Embedding(len(word2ix), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.n_tags)  # 用于将LSTM的输出 降维到 标签空间
        
        self.transitions = nn.Parameter(torch.randn(self.n_tags, self.n_tags))

        self.transitions.data[tag2ix[START_TAG], :] = self.transitions.data[:, tag2ix[END_TAG]] = -10000

        self.hidden = self.init_hidden()

    def neg_log_likelihood(self, words, tags):  # 求一对 <sentence, tags> 在当前参数下的负对数似然，作为loss
        frames = self._get_lstm_features(words)  # emission score at each frame
        gold_score = self._score_sentence(frames, tags)  # 正确路径的分数
        forward_score = self._forward_alg(frames)  # 所有路径的分数和
        # -(正确路径的分数 - 所有路径的分数和）;注意取负号 -log(a/b) = -[log(a) - log(b)] = log(b) - log(a)
        return forward_score - gold_score

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, words):  # 求出每一帧对应的隐向量
        # LSTM输入形状(seq_len, batch=1, input_size); 教学演示 batch size 为1
        #print(self._to_tensor(words, self.word2ix))
        embeds = self.word_embeds(self._to_tensor(words, self.word2ix)).view(len(words), 1, -1)
        #print(embeds)
        
        # 随机初始化LSTM的隐状态H
        self.hidden = self.init_hidden()
        #print('\n------------hidden---------')
        #print(hidden)
        lstm_out, _hidden = self.lstm(embeds, self.hidden)
        #print(lstm_out)
        return self.hidden2tag(lstm_out.squeeze(1))  # 把LSTM输出的隐状态张量去掉batch维，然后降维到tag空间

    def _score_sentence(self, frames, tags):
        """
        求路径pair: frames->tags 的分值
        index:      0   1   2   3   4   5   6
        frames:     F0  F1  F2  F3  F4
        tags:  <s>  Y0  Y1  Y2  Y3  Y4  <e>
        """
        tags_tensor = self._to_tensor([START_TAG] + tags, self.tag2ix)  # 注意不要+[END_TAG]; 结尾有处理
        score = torch.zeros(1)
        for i, frame in enumerate(frames):  # 沿途累加每一帧的转移和发射
            score += self.transitions[tags_tensor[i + 1], tags_tensor[i]] + frame[tags_tensor[i + 1]]
        return score + self.transitions[self.tag2ix[END_TAG], tags_tensor[-1]]  # 加上到END_TAG的转移

    def _forward_alg(self, frames):
        """ 给定每一帧的发射分值; 按照当前的CRF层参数算出所有可能序列的分值和，用作概率归一化分母 """
        alpha = torch.full((1, self.n_tags), -10000.0)
        alpha[0][self.tag2ix[START_TAG]] = 0  # 初始化分值分布. START_TAG是log(1)=0, 其他都是很小的值 "-10000"
        for frame in frames:
            # log_sum_exp()内三者相加会广播: 当前各状态的分值分布(列向量) + 发射分值(行向量) + 转移矩阵(方形矩阵)
            # 相加所得矩阵的物理意义见log_sum_exp()函数的注释; 然后按列求log_sum_exp得到行向量
            alpha = log_sum_exp(alpha.T + frame.unsqueeze(0) + self.transitions.T)
        # 最后转到EOS，发射分值为0，转移分值为列向量 self.transitions[:, [self.tag2ix[END_TAG]]]
        return log_sum_exp(alpha.T + 0 + self.transitions[[self.tag2ix[END_TAG]], :].T).flatten()

    def _viterbi_decode(self, frames):
        backtrace = []  # 回溯路径;  backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是神马状态
        alpha = torch.full((1, self.n_tags), -10000.)
        alpha[0][self.tag2ix[START_TAG]] = 0
        for frame in frames:
            # 这里跟 _forward_alg()稍有不同: 需要求最优路径（而非一个总体分值）, 所以还要对smat求column_max
            smat = alpha.T + frame.unsqueeze(0) + self.transitions.T
            
            val, idx = torch.max(smat, 0)
            backtrace.append(idx)
            alpha = val.unsqueeze(0)
            
            #backtrace.append(smat.argmax(0))  # 当前帧每个状态的最优"来源"
            #alpha = log_sum_exp(smat)  # 转移规律跟 _forward_alg()一样; 只不过转移之前拿smat求了一下回溯路径

        # 回溯路径
        smat = alpha.T + 0 + self.transitions[[self.tag2ix[END_TAG]], :].T
        
        val, idx = torch.max(smat, 0)
        best_tag_id = idx.item()
              
        #best_tag_id = smat.flatten().argmax().item()
        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):  # 从[1:]开始，去掉开头的 START_TAG
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        return val.item(), best_path[::-1]  # 返回最优路径分值 和 最优路径

    def forward(self, words):  # 模型inference逻辑
        lstm_feats = self._get_lstm_features(words)  # 求出每一帧的发射矩阵
        #print(lstm_feats)
        return self._viterbi_decode(lstm_feats)  # 采用已经训好的CRF层, 做维特比解码, 得到最优路径及其分数

    def _to_tensor(self, words, to_ix):  # 将words/tags序列数值化，即: 映射为相应下标序列张量
        return torch.tensor([to_ix[w] for w in words], dtype=torch.long)

if __name__ == "__main__":
    training_data = [("the wall street journal reported today that apple corporation made money".split(),
                      "B I I I O O O B I O O".split()),
                     ("georgia tech is a university in georgia".split(), "B I O O O O B".split())]

    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    model = BiLSTM_CRF(tag2ix={"B": 0, "I": 1, "O": 2, START_TAG: 3, END_TAG: 4},
                       word2ix=word_to_ix,
                       embedding_dim=5, hidden_dim=4)

    with torch.no_grad():  # 训练前, 观察一下预测结果(应该是随机或者全零参数导致的结果)
        print(model(training_data[0][0]))

    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    for epoch in range(300):  # 不要试图改成100, 在这个教学例子数据集上会欠拟合……
        for words, tags in training_data:
            model.zero_grad()  # PyTorch默认会累积梯度; 而我们需要每条样本单独算梯度
            model.neg_log_likelihood(words, tags).backward()  # 前向求出负对数似然(loss); 然后回传梯度
            optimizer.step()  # 梯度下降，更新参数
    # 训练后的预测结果(有意义的结果，与label一致); 打印类似 (18.722553253173828, [0, 1, 1, 1, 2, 2, 2, 0, 1, 2, 2])
    with torch.no_grad():  # 这里用了第一条训练数据(而非专门的测试数据)，仅作教学演示
        print(model(training_data[0][0]))