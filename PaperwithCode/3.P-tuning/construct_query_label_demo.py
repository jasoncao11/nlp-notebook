# -*- coding: utf-8 -*-
import torch
from torch.nn.utils.rnn import pad_sequence

cls_token_id = 102
sep_token_id = 103
mask_token_id = 2
pseudo_token_id = 1
unk_token_id = 3
template = (2,2,2)
x_h_1 = 90
x_h_2 = 80
x_t_1 = 100
x_t_2 = 200
batch_size = 2

queries = [torch.LongTensor([cls_token_id,pseudo_token_id,pseudo_token_id,mask_token_id,pseudo_token_id,pseudo_token_id,x_h_1,pseudo_token_id,pseudo_token_id,sep_token_id]), 
           torch.LongTensor([cls_token_id,pseudo_token_id,pseudo_token_id,mask_token_id,pseudo_token_id,pseudo_token_id,x_h_2,pseudo_token_id,pseudo_token_id,sep_token_id])]
#print(queries)
queries = pad_sequence(queries, True, padding_value=0).long()
print(queries)

queries_for_embedding = queries.clone()
queries_for_embedding[(queries == pseudo_token_id)] = unk_token_id
print(queries_for_embedding)
#raw_embeds = embeddings(queries_for_embedding)
print('-------------------------------------------')
print((queries == pseudo_token_id))
print((queries == pseudo_token_id).nonzero())
print((queries == pseudo_token_id).nonzero().reshape((batch_size, sum(template), 2)))
blocked_indices = (queries == 1).nonzero().reshape((batch_size, sum(template), 2))[:, :, 1]
print(blocked_indices)

#根据每个BATCH中为pseudo_token_id的索引，使用prompt_encoder的结果进行替代
#replace_embeds = prompt_encoder()
#for bidx in range(bz):
#    for i in range(self.prompt_encoder.spell_length):
#        raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]

print('-------------------------------------------')
print((queries == mask_token_id))
print((queries == mask_token_id).nonzero())
print((queries == mask_token_id).nonzero().reshape(batch_size, -1))
print((queries == mask_token_id).nonzero().reshape(batch_size, -1)[:, 1])
label_mask = (queries == mask_token_id).nonzero().reshape(batch_size, -1)[:, 1].unsqueeze(1)
print(label_mask)

labels = torch.empty_like(queries).fill_(-100).long()
print(labels)

label_ids = torch.LongTensor([x_t_1, x_t_2]).reshape((batch_size, -1))
print(label_ids)
labels = labels.scatter_(1, label_mask, label_ids)
print(labels)