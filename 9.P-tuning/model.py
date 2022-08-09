# -*- coding: utf-8 -*-
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertForMaskedLM
from prompt_encoder import PromptEncoder

class PTuneForLAMA(torch.nn.Module):

    def __init__(self, device, template):
        super().__init__()
        self.device = device

        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # load pre-trained model
        self.model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.model = self.model.to(self.device)
        for param in self.model.parameters():
            param.requires_grad = False
        self.embeddings = self.model.get_input_embeddings()

        # set allowed vocab set
        self.vocab = self.tokenizer.get_vocab()

        self.template = template

        # load prompt encoder
        self.hidden_size = self.embeddings.embedding_dim
        self.tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]']})
        self.pseudo_token_id = self.tokenizer.get_vocab()['[PROMPT]']
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.unk_token_id

        self.spell_length = sum(self.template)
        self.prompt_encoder = PromptEncoder(self.template, self.hidden_size, self.device)
        self.prompt_encoder = self.prompt_encoder.to(self.device)

    def embed_input(self, queries):
        bz = queries.shape[0]
        queries_for_embedding = queries.clone()
        queries_for_embedding[(queries == self.pseudo_token_id)] = self.tokenizer.unk_token_id
        raw_embeds = self.embeddings(queries_for_embedding) #[batch size, spell_length + x, hidden_size]

        blocked_indices = (queries == self.pseudo_token_id).nonzero().reshape((bz, self.spell_length, 2))[:, :, 1] #[batch size, spell_length] 找出每个BATCH中为prompt的索引，用于之后使用prompt_encoder的结果进行替代
        replace_embeds = self.prompt_encoder() #[spell_length, hidden_size] 
        for bidx in range(bz):
            for i in range(self.spell_length):
                raw_embeds[bidx, blocked_indices[bidx, i], :] = replace_embeds[i, :]
        return raw_embeds #[batch size, spell_length + x, hidden_size]

    def get_query(self, x_h, prompt_tokens, x_t=None):
        # For P-tuning
        # BERT-style model
        return [[self.tokenizer.cls_token_id]
                + prompt_tokens * self.template[0]
                + [self.tokenizer.mask_token_id]
                + prompt_tokens * self.template[1]
                + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(x_h))
                + prompt_tokens * self.template[2]
                + [self.tokenizer.sep_token_id]]
       
    def forward(self, x_hs, x_ts):
        bz = len(x_hs)

        # construct query ids
        prompt_tokens = [self.pseudo_token_id]
       
        queries = [torch.LongTensor(self.get_query(x_hs[i], prompt_tokens)).squeeze(0) for i in range(bz)] #[batch size, spell_length + x]
        queries = pad_sequence(queries, True, padding_value=self.pad_token_id).long().to(self.device)

        # construct label ids
        label_ids = torch.LongTensor(self.tokenizer.convert_tokens_to_ids(x_ts)).reshape((bz, -1)).to(self.device)
        attention_mask = queries != self.pad_token_id
        #print(attention_mask)

        # get embedded input
        inputs_embeds = self.embed_input(queries) #[batch size, spell_length + x, hidden_size]

        label_mask = (queries == self.tokenizer.mask_token_id).nonzero().reshape(bz, -1)[:, 1].unsqueeze(1).to(self.device)  # bz * 1
        labels = torch.empty_like(queries).fill_(-100).long().to(self.device)  # bz * seq_len
        labels = labels.scatter_(1, label_mask, label_ids)
        output = self.model(inputs_embeds=inputs_embeds.to(self.device),
                            attention_mask=attention_mask.to(self.device).bool(),
                            labels=labels.to(self.device))
        return output.loss