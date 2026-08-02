import torch

from data.dataloader import dataloader
from attention.multi_head_attention import MultiHeadAttention
from model.gpt import GPTModel
from utils.model_config import GPT_CONFIG_124M
from inference.generate import generate_text
import tiktoken


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    token_ids_list = token_ids.squeeze(0).tolist()  # Remove batch dimension and convert to list
    decoded_text = tokenizer.decode(token_ids_list)
    return decoded_text


token_ids = generate_text(
    model=GPTModel(GPT_CONFIG_124M),
    idx=text_to_token_ids("Once upon a time", tiktoken.get_encoding("gpt2")),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)

#print("Output text:\n", token_ids_to_text(token_ids, tiktoken.get_encoding("gpt2")))

model = GPTModel(GPT_CONFIG_124M)

inputs = torch.tensor([[16833, 3626, 6100],
                       [40, 1107, 588]])

targets = torch.tensor([[3626, 6100, 345],
                       [1107, 588, 11311]])

with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim=-1)
token_ids = torch.argmax(probas, dim=-1, keepdim=True)

#print(f"Targets batch 1: {token_ids_to_text(targets[0], tiktoken.get_encoding('gpt2'))}")
#print(f"Generated batch 1: {token_ids_to_text(token_ids[0].flatten(), tiktoken.get_encoding('gpt2'))}")

text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
#print("Text 1:", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
#print("Text 2:", target_probas_2)

log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
#print(log_probas)

avg_log_proba = torch.mean(log_probas)
print(avg_log_proba)

logits_flat = logits.flatten(0,1)
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)

loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print("Loss:", loss)