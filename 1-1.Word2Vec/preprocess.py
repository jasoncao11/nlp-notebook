# -*- coding: utf-8 -*-
import torch
import torch.utils.data as tud

C = 2 # context window size
K = 15 # number of negative samples, K is approximate to C*2*5 for middle size corpus, thst is to pick 5 negative samples for each context word selected

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word2idx, word_freqs):
        '''
        Args:
            text: list of all the words from the training dataset
            word2idx: the mapping from word to index
            word_freqs: normalized frequency of each word
        '''
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word2idx.get(word, word2idx['<UNK>']) for word in text]
        self.word_freqs = torch.Tensor(word_freqs)
        self.word2idx = word2idx
               
    def __len__(self):
        return len(self.text_encoded)
    
    def __getitem__(self, idx):
        ''' 
        return:
            - center word index
            - C indices of positive words
            - K indices of negative words
        '''
        center_words = torch.LongTensor(self.text_encoded)[idx]
        left = self.text_encoded[max(idx - C, 0) : idx]
        right = self.text_encoded[idx + 1 : idx + 1 + C]
        pos_words = [self.word2idx['<UNK>'] for _ in range(C - len(left))] + left + right + [self.word2idx['<UNK>'] for _ in range(C - len(right))]        
        neg_words = torch.multinomial(self.word_freqs, K, True)
        return center_words, torch.LongTensor(pos_words), neg_words