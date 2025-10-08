import torch

class LMWindowDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, context_len):
        self.toks = tokens
        self.ctx = context_len
    def __len__(self):
        return len(self.toks) - self.ctx # number of windows
    def __getitem__(self, i): # inputs:targets
        x = torch.tensor(self.toks[i:i+self.ctx], dtype=torch.long)
        y = torch.tensor(self.toks[i+1:i+self.ctx+1], dtype=torch.long)
        return x, y