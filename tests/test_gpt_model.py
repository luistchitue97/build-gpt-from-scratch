import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model.gpt import GPTModel


def test_gpt_model_processes_batched_sequences():
    cfg = {
        "vocab_size": 20,
        "context_length": 32,
        "emb_dim": 16,
        "n_heads": 4,
        "n_layers": 1,
        "dropout_rate": 0.0,
        "qkv_bias": False,
    }

    model = GPTModel(cfg)
    batch = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)

    with torch.no_grad():
        logits = model(batch)

    assert logits.shape == (2, 4, 20)
