import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout_rate: float = 0.0, n_expand: int = 4) -> None:
        super().__init__()
        self.hidden_layer = nn.Linear(in_features=dim, out_features=n_expand*dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(in_features=n_expand*dim, out_features=dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x