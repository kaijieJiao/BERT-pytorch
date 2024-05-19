import torch
import torch.nn as nn
import json

class BERTVocab(object):
    def __init__(self) -> None:
        super.__init__()
        self.UNK_TOKEN = '<UNK>'
        self.PAD_TOKEN = '<PAD>'
        self.SOS_TOKEN = '<SOS>'
        self.EOS_TOKEN = '<EOS>'
        self.MASK_TOKEN = '<MASK>'
        self.specials_tokens = [self.UNK_TOKEN, self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN, self.MASK_TOKEN]
        self.UNK_ID = 0
        self.PAD_ID = 1
        self.SOS_ID = 2
        self.EOS_ID = 3
        self.MASK_ID = 4
        self.itos = {}
        self.stoi = {}
        for i,token in enumerate(self.specials_tokens):
            self.itos[i] = token
            self.stoi[token]=i
    def load_vocab(self,vocab_path: str):
        self.itos = json.load(open(vocab_path))
        self.stoi = {tok: i for i, tok in self.itos.items()}
        return self.stoi
    def save_vocab(self, vocab_path):
        json.dump(self.itos, open(vocab_path, "w"))
    
    def add_tokens(self,tokens):
        for token in tokens:
            if token not in self.stoi:
                self.itos[len(self.itos)] = token
                self.stoi[token] = len(self.stoi)
    def __len__(self):
        return len(self.itos)

    def sentence_to_ids(self, sentence):
        return [self.stoi.get(token,self.UNK_ID) for token in sentence.split()]