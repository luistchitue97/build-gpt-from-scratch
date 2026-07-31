import torch.nn as nn

import attention.multi_head_attention as mha
from layers.layer_norm import LayerNorm
from layers.feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = mha.MultiHeadAttention(
            in_dim=cfg["emb_dim"],
            out_dim=cfg["emb_dim"],
            cont_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["dropout_rate"],
            qkv_bias=cfg["qkv_bias"]
        )
        self.layer_norm1 = LayerNorm(cfg["emb_dim"])
        self.feed_forward = FeedForward(cfg)
        self.layer_norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["dropout_rate"])
        
    def forward(self, x):
        shortcut = x
        x = self.layer_norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x) 
        x = x + shortcut

        return x