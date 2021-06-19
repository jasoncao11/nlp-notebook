# -*- coding: utf-8 -*-
import numpy as np
import torch
from transformers import BertModel, BertTokenizer

class BertParent(object):
    """
    Base handler for BERT models.
    """
    def __init__(self, bert_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.model = BertModel.from_pretrained(bert_path, output_hidden_states=True).to(self.device)
        self.model.eval()

    def tokenize_input(self, text):
        """
        Tokenizes the text input.
        :param text: Text to tokenize.
        :return: Returns a torch tensor.
        """
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        return torch.tensor([indexed_tokens]).to(self.device)

    def extract_embeddings(self, text):
        """
        Extracts the embeddings for the given text.
        :param text: The text to extract embeddings for.
        :return: A torch vector.
        """
        tokens_tensor = self.tokenize_input(text)
        pooled, hidden_states = self.model(tokens_tensor)[-2:]
        last_4 = [hidden_states[i] for i in (-1, -2, -3, -4)]
        cat_hidden_states = torch.cat(tuple(last_4), dim=-1)
        return torch.mean(cat_hidden_states, dim=1).squeeze()    

    def create_matrix(self, content):
        """
        Create matrix from the embeddings.
        :param content: The list of sentences.
        :return: A numpy array matrix of the given content.
        """
        return np.asarray([np.squeeze(self.extract_embeddings(t).data.cpu().numpy()) for t in content])

    def __call__(self, content):
        """
        Create matrix from the embeddings.
        :param content: The list of sentences.
        :return: A numpy array matrix of the given content.
        """
        return self.create_matrix(content)