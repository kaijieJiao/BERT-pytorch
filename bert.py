import torch
import torch.nn as nn
from bertblock import BERTBlock
from embedding import BERTEmbedding
class BERT(nn.Module):
    def __init__(self, embedding_size,vocab_size ,  hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        self.embedding = BERTEmbedding(vocab_size=vocab_size,embedding_size=embedding_size)
        self.bert = nn.ModuleList([BERTBlock(embedding_size,hidden, attn_heads, dropout) for _ in range(n_layers)])
        self.mask_out = nn.Linear(hidden, vocab_size)
        self.nsp_out = nn.Linear(hidden, 2)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, segment_label,mask):
        x = self.embedding(x, segment_label)
        for i in range(self.n_layers):
            x = self.bert[i](x, mask)
        return self.dropout(self.nsp_out(x)),self.dropout(self.mask_out(x))
        
