import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim, out_dim, cont_length, num_heads, qkv_bias=False, dropout=0.1):
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads


        # dropout layer for attention weights
        self.attn_dropout = nn.Dropout(dropout)

        # projection matrices:
        self.W_key = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_query = nn.Linear(in_dim, out_dim, bias=qkv_bias)
        self.W_value = nn.Linear(in_dim, out_dim, bias=qkv_bias)        

        # mask for attention to prevent attending to future tokens
        self.register_buffer("mask", torch.triu(torch.ones(cont_length, cont_length), diagonal=1))

        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        b, cont_length, d_in = x.size()

        # projections
        queries = self.W_query(x)
        keys = self.W_key(x) 
        values = self.W_value(x)

        queries = queries.view(b, cont_length, self.num_heads, self.head_dim)
        keys = keys.view(b, cont_length, self.num_heads, self.head_dim)
        values = values.view(b, cont_length, self.num_heads, self.head_dim)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(self.mask[:cont_length, :cont_length] == 1, -torch.inf)

        #attn_weights = attn_weights.masked_fill(self.mask[:cont_length, :cont_length] == 1, float('-inf'))
        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2).contiguous().view(b, cont_length, self.out_dim)

        context_vec = self.out_proj(context_vec)
        return context_vec