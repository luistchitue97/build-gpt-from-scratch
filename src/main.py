import torch

from data.dataloader import dataloader
from attention.multi_head_attention import MultiHeadAttention
from model.gpt import GPTModel
from utils.model_config import GPT_CONFIG_124M


#iterator = iter(dataloader())
model = GPTModel(GPT_CONFIG_124M)
#inputs, targets = next(iterator)
#print("Inputs:", inputs)
#print("Targets:", targets)


inputs = torch.tensor(
    [[1, 2, 3, 4]],
    dtype=torch.long,
)

out = model(inputs)
print("Input batch:\n", inputs)
print("Output shape:", out.shape)
print(out)
