import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken

class Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.inputs = []

        tokenizer = tiktoken.get_encoding("gpt2")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, key):
        return []
