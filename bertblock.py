import torch
import torch.nn as nn
from multiheadattention import MultiHeadAttention

class BERTBlock(nn.Module):
    def __init__(self,embedding_size,hidden, attn_heads, feed_forward_hidden, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(embedding_size,hidden, attn_heads,dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, hidden),
        )
        self.laynorm1 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.laynorm2 = nn.LayerNorm(hidden)
    
    def forward(self,x,mask):
        y = self.attention(x,mask)
        z = self.laynorm1(x + y)
        output = self.laynorm2(z + self.feed_forward(z))
        return output