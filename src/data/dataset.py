import torch
from torch.utils.data import Dataset
import tiktoken

class Dataset(Dataset):
    def __init__(self, txt, context_length, stride):
        super().__init__()
        self.inputs = []
        self.targets = []

        tokenizer = tiktoken.get_encoding("gpt2")
        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - context_length, stride):
            input_ids = token_ids[i:i + context_length]
            target_ids = token_ids[i + 1:i + context_length + 1]   

            self.inputs.append(torch.tensor(input_ids, dtype=torch.long))
            self.targets.append(torch.tensor(target_ids, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, key):
        return self.inputs[key], self.targets[key]
