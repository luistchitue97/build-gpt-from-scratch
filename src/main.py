import torch

from data.dataloader import dataloader
from attention.multi_head_attention import MultiHeadAttention

iterator = iter(dataloader())
inputs, targets = next(iterator)
print("Inputs:", inputs)
print("Targets:", targets)


inputs = torch.tensor(
  [[0.43, 0.15, 0.89], 
   [0.55, 0.87, 0.66], 
   [0.57, 0.85, 0.64], 
   [0.22, 0.58, 0.33], 
   [0.77, 0.25, 0.10], 
   [0.05, 0.80, 0.55]] 
)

batch = torch.stack([inputs, inputs], dim=0)  # Create a batch of size 2

batch_size, cont_length, in_dim = batch.shape
d_out = 4
mha = MultiHeadAttention(in_dim=in_dim, out_dim=d_out, cont_length=cont_length, num_heads=2, dropout=0.0)
context_vec = mha(batch)    
print("Context Vector:", context_vec)
print("Context Vector Shape:", context_vec.shape)