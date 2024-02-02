import torch
import torch.nn as nn

from model.utils.masking import generate_look_ahead_mask, generate_mask
from model.modules.encoder import Encoder
from model.modules.decoder import Decoder

from typing import Optional

class BART(nn.Module):
    def __init__(self, token_size: int, n_layers: int, d_model: int, heads: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.encoder = Encoder(token_size=token_size, n_layers=n_layers, d_model=d_model, heads=heads, dropout_rate=dropout_rate)
        self.decoder = Decoder(token_size=token_size, n_layers=n_layers, d_model=d_model, heads=heads, dropout_rate=dropout_rate)

        self.head = nn.Linear(in_features=d_model, out_features=token_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, y_lengths: Optional[torch.Tensor] = None):
        padding_mask = None
        look_ahead_mask = None
        if x_lengths is not None and y_lengths is not None:
            padding_mask = generate_mask(x_lengths)
            look_ahead_mask = generate_look_ahead_mask(y_lengths)

        encoder_output = self.encoder(x, padding_mask)
        decoder_output = self.decoder(y, encoder_output, look_ahead_mask, padding_mask)

        output = self.head(decoder_output)

        return output
    
    def infer(self, x: torch.Tensor, y: torch.Tensor, max_steps: int, end_token: int):
        encoder_output = self.encoder(x)

        for _ in range(max_steps):
            output = self.head(self.decoder(y, encoder_output))
            pred = torch.argmax(output[:, -1, :], dim=-1)

            if pred == end_token:
                break

            y = torch.concat([y, pred], dim=-1)

        return y