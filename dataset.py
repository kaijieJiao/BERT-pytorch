import torch
import torch.nn as nn
from torch.utils.data import Dataset
from vocab import BERTVocab
class BERTDataset(Dataset):
    
    def __init__(self,sentences) -> None:
        super().__init__()
        vocab = BERTVocab()
        pass
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        bert_input ={
            'input_ids': torch.tensor(self.data[index]['input_ids']),
            'sgement_ids': torch.tensor(self.data[index]['sgement_ids']),
            'attention_mask': torch.tensor(self.data[index]['attention_mask']),
            'labels': torch.tensor(self.data[index]['labels'])
        }
        return bert_input