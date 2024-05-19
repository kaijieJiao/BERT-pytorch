import torch
import torch.nn as nn

class BERTEmbedding(nn.Module):
    def __init__(self,vocab_size,embedding_size,seq_max_len=2048,dropout=0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.token_embedding = nn.Embedding(vocab_size,embedding_size)
        self.position_embedding = nn.Embedding(seq_max_len,embedding_size)
        self.segment_embedding = nn.Embedding(3,embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,segment_label):
        token_emb = self.token_embedding(x)
        position_emb = self.position_embedding(torch.arange(x.size(1)))
        segment_emb = self.segment_embedding(segment_label)
        return self.dropout(token_emb + position_emb + segment_emb)

if __name__ == '__main__':
    ids = torch.tensor([[1,2,3,4,5,0,6,7,8,9,10],[6,7,8,9,10,0,1,2,3,4,5]])
    segment_label = torch.tensor([[0,0,0,0,0,1,1,1,1,1,1],[1,1,1,1,1,1,0,0,0,0,0]])
    embedding = BERTEmbedding(1000,128)
    print(embedding(ids,segment_label).shape)