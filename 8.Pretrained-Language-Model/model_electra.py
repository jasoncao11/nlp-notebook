import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else 'cpu'

def log(t, eps=1e-9):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1.):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

class Electra(nn.Module):

    def __init__(self, generator, discriminator, gen_weight, disc_weight):
        super(Electra, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.gen_weight = gen_weight
        self.disc_weight = disc_weight

    def simcse_unsup_loss(self, y_pred):
        """无监督的损失函数
        y_pred (tensor): bert的输出, [batch_size, 768]
        """
        # 得到y_pred对应的label, [1, 0, 3, 2, ..., batch_size-1, batch_size-2]
        y_true = torch.arange(y_pred.shape[0], device=device)
        y_true = (y_true - y_true % 2 * 2) + 1
        # batch内两两计算相似度, 得到相似度矩阵(对角矩阵)
        sim = F.cosine_similarity(y_pred.unsqueeze(1), y_pred.unsqueeze(0), dim=-1)
        # 将相似度矩阵对角线置为很小的值, 消除自身的影响
        sim = sim - torch.eye(y_pred.shape[0], device=device) * 1e12
        # 相似度矩阵除以温度系数
        sim = sim / 0.05
        # 计算相似度矩阵与y_true的交叉熵损失
        loss = F.cross_entropy(sim, y_true)
        return loss

    def forward(self, original_input_ids, input_ids, labels, attention_mask):
        #mlm loss for generator
        logits = self.generator(input_ids=input_ids, attention_mask=attention_mask)
        gen_loss_fct = CrossEntropyLoss() #Tokens with indices set to ``-100`` are ignored
        masked_lm_loss = gen_loss_fct(logits.view(-1, 21128), labels.view(-1))

        #gumbel_sample
        pred_indices = torch.nonzero(labels!=-100, as_tuple=True)
        sample_logits = logits[pred_indices]
        sampled = gumbel_sample(sample_logits, temperature = 1.)
        
        #loss for discriminator
        disc_input = original_input_ids.clone()
        disc_input[pred_indices] = sampled.detach()
        #generate discriminator labels, with replaced as True and original as False
        disc_labels = (original_input_ids != disc_input).float().detach()
        disc_logits, pooler = self.discriminator(input_ids=disc_input, attention_mask=attention_mask)
        disc_loss_fct = BCEWithLogitsLoss()
        disc_loss = disc_loss_fct(disc_logits, disc_labels)
        simcse_loss = self.simcse_unsup_loss(pooler)
        #total loss       
        loss = self.gen_weight * masked_lm_loss + self.disc_weight * disc_loss + simcse_loss
        return loss