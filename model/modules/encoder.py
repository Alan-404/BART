import torch
import torch.nn as nn
from model.utils.position import PositionalEncoding
from model.utils.block import EncoderBlock

from typing import Optional

class Encoder(nn.Module):
    def __init__(self, token_size: int, n_layers: int, d_model: int, heads: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=token_size, embedding_dim=d_model)
        self.pe = PositionalEncoding(d_model=d_model)
        self.layers = nn.ModuleList([EncoderBlock(d_model=d_model, heads=heads, dropout_rate=dropout_rate) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(normalized_shape=d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        x = self.embedding(x)
        x = x + self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        return x