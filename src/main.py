import torch

from data.dataloader import dataloader
from attention.multi_head_attention import MultiHeadAttention
from model.gpt import GPTModel
from utils.model_config import GPT_CONFIG_124M
from inference.generate import generate_text
import tiktoken


#iterator = iter(dataloader())
model = GPTModel(GPT_CONFIG_124M)
#inputs, targets = next(iterator)
#print("Inputs:", inputs)
#print("Targets:", targets)


#batch = torch.tensor(
#    [[1, 2, 3, 4],
#     [2, 3, 4, 5]],
#    dtype=torch.long,
#)

#out = model(batch)
#print("Input batch shape:", batch.shape)
#print("Output shape:", out.shape)
#print(out)

start_context = "Hello, I am"
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context)

print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)  # Add batch dimension
print("encoded_tensor shape:", encoded_tensor.shape)

model.eval()
out = generate_text(
    model=model, 
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Generated output:", out)
print("Output length:", len(out[0]))

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)