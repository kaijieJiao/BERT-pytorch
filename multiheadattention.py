import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size,hidden, n_heads, dropout):
        super().__init__()
        self.hidden = hidden
        self.n_heads = n_heads
        self.dropout = dropout
        # q、k、v相同size
        self.query_linear = nn.Linear(embedding_size, hidden)
        self.key_linear = nn.Linear(embedding_size, hidden)
        self.value_linear = nn.Linear(embedding_size, hidden)
        self.out = nn.Linear(hidden, embedding_size)

    def forward(self,x,mask):
        q=self.query_linear(x).view(x.size(0),x.size(1),self.n_heads,-1).transpose(1,2)
        k=self.key_linear(x).view(x.size(0),x.size(1),self.n_heads,-1).transpose(1,2).transpose(2,3)
        v=self.value_linear(x).view(x.size(0),x.size(1),self.n_heads,-1).transpose(1,2)
        scores = torch.matmul(q,k)/math.sqrt(k.size(-1))
        scores=scores.masked_fill_(mask, -1e9)
        scores=F.softmax(scores,dim=-1)
        attention = torch.matmul(scores,v).transpose(2,3).view(x.size(0),x.size(1),-1)
        attention=self.dropout(self.out(attention))
        return attention