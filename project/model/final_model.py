import torch
import torch.nn as nn
from model.transformer_block import *

class miniLLM(nn.Module):
    def __init__(self, vocab_size, d_model=512, num_heads=8, d_ff=2048, num_layers=6, context_len=256, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerDecoderOnlyBlock(num_heads, d_model, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = NormLayer(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape

        pos = torch.arange(T, device=idx.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(idx) * math.sqrt(self.tok_emb.embedding_dim)
        x = x + self.pos_emb(pos)

        #x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        logits = self.lm_head(x)  
        return logits