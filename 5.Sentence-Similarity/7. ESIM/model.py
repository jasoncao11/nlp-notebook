import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMEncoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)

    def forward(self, sequences_batch, sequences_lengths):
        """
        参考：https://zhuanlan.zhihu.com/p/342685890
        Args:
            sequences_batch: A batch of variable length sequences of vectors.
                The batch is assumed to be of size
                (batch, sequence, input_size).
            sequences_lengths: A 1D tensor containing the sizes of the
                sequences in the input batch.
        Returns:
            restored_outputs: The outputs (hidden states) of the encoder for
                the sequences in the input batch, in the same order.
        """
        #按长度降序排序
        sorted_lengths, sorted_index = sequences_lengths.sort(0, descending=True)
        sorted_batch = sequences_batch.index_select(0, sorted_index)
        _, restored_index = sorted_index.sort(0, descending=False)       
        #先pack，经过RNN处理后再pad   
        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch, sorted_lengths.to('cpu'), batch_first=True)
        outputs, _ = self.encoder(packed_batch, None)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        restored_outputs = outputs.index_select(0, restored_index)
        return restored_outputs #(batch, sequence, hidden_size*2)
    
class SoftmaxAttention(nn.Module):

    def forward(self, sentences_a_batch, sentences_a_mask, sentences_b_batch, sentences_b_mask):
        # sentences_a_batch: (batch, seq_a, vector_dim)
        # sentences_a_mask: (batch, seq_a)
        # sentences_b_batch: (batch, seq_b, vector_dim)
        # sentences_b_mask: (batch, seq_b)
        
        # Dot product between sentences_a_batch and sentences_b_batch
        similarity_matrix = sentences_a_batch.bmm(sentences_b_batch.transpose(2, 1)) #(batch, seq_a, seq_b)
        
        # Attention mask
        extended_a_mask = sentences_a_mask.unsqueeze(1) #(batch, 1, seq_a)
        a_mask_1 = (1.0 - extended_a_mask) * -10000.0
        extended_b_mask = sentences_b_mask.unsqueeze(1) #(batch, 1, seq_b)
        b_mask_1 = (1.0 - extended_b_mask) * -10000.0
        
        # Softmax attention weights.
        a_b_attn = nn.Softmax(dim=-1)(similarity_matrix + b_mask_1) #(batch, seq_a, seq_b)
        b_a_attn = nn.Softmax(dim=-1)(similarity_matrix.transpose(1, 2) + a_mask_1) #(batch, seq_b, seq_a)
        
        # Weighted sums
        attended_a = a_b_attn.bmm(sentences_b_batch) #(batch, seq_a, vector_dim)
        attended_b = b_a_attn.bmm(sentences_a_batch) #(batch, seq_b, vector_dim)

        return attended_a, attended_b
    
class ESIM(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 dropout=0.5,
                 num_classes=2):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 2.
        """
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout

        self.word_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.encoding = LSTMEncoder(self.embedding_dim, self.hidden_size)
        self.attention = SoftmaxAttention()

        self.projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,self.hidden_size),
                                        nn.ReLU())

        self.composition = LSTMEncoder(self.hidden_size, self.hidden_size)

        self.classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                            nn.Linear(2*4*self.hidden_size, self.hidden_size),
                                            nn.Tanh(),
                                            nn.Dropout(p=self.dropout),
                                            nn.Linear(self.hidden_size, self.num_classes))

    def forward(self,sentences_a_batch, sentences_a_mask, sentences_a_length,
                     sentences_b_batch, sentences_b_mask, sentences_b_length):
        
        # sentences_a_batch: (batch, seq_a)
        # sentences_a_mask: (batch, seq_a)
        # sentences_a_length: (batch,)
        # sentences_b_batch: (batch, seq_b)
        # sentences_b_mask: (batch, seq_b)
        # sentences_b_length: (batch,)
 
        embedded_a = self.word_embedding(sentences_a_batch) #(batch, seq_a, embedding_dim)
        embedded_b = self.word_embedding(sentences_b_batch) #(batch, seq_b, embedding_dim)

        encoded_a = self.encoding(embedded_a, sentences_a_length) #(batch, seq_a, hidden_size*2)
        encoded_b = self.encoding(embedded_b, sentences_b_length) #(batch, seq_b, hidden_size*2)
        
        #attended_a: (batch, seq_a, hidden_size*2)
        #attended_b: (batch, seq_b, hidden_size*2)
        attended_a, attended_b = self.attention(encoded_a, sentences_a_mask, encoded_b, sentences_b_mask)
        
        enhanced_a = torch.cat([encoded_a, attended_a, encoded_a - attended_a, encoded_a * attended_a],dim=-1) #(batch, seq_a, hidden_size*2*4) 
        enhanced_b = torch.cat([encoded_b, attended_b, encoded_b - attended_b, encoded_b * attended_b],dim=-1) #(batch, seq_b, hidden_size*2*4)
        
        projected_a = self.projection(enhanced_a) #(batch, seq_a, hidden_size) 
        projected_b = self.projection(enhanced_b) #(batch, seq_b, hidden_size)    
        
        v_ai = self.composition(projected_a, sentences_a_length) #(batch, seq_a, hidden_size*2) 
        v_bj = self.composition(projected_b, sentences_b_length) #(batch, seq_b, hidden_size*2)        
        
        v_a_avg = torch.sum(v_ai * sentences_a_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(sentences_a_mask, dim=1, keepdim=True) #(batch, hidden_size*2) 
        v_b_avg = torch.sum(v_bj * sentences_b_mask.unsqueeze(1).transpose(2, 1), dim=1) / torch.sum(sentences_b_mask, dim=1, keepdim=True) #(batch, hidden_size*2)  
    
        extended_a_mask = (1.0 - sentences_a_mask.unsqueeze(1).transpose(2, 1)) * -10000.0 #(batch, seq_a, 1)
        v_a_max = (v_ai + extended_a_mask).max(dim=1).values #(batch, hidden_size*2) 

        extended_b_mask = (1.0 - sentences_b_mask.unsqueeze(1).transpose(2, 1)) * -10000.0 #(batch, seq_b, 1)
        v_b_max = (v_bj + extended_b_mask).max(dim=1).values #(batch, hidden_size*2) 
        
        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self.classification(v) #(batch, num_classes)
        return logits