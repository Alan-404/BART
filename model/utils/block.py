import torch
import torch.nn as nn

from model.utils.attention import MultiHeadAttention
from model.utils.ffn import FeedForward

from typing import Optional

class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model=d_model, heads=heads, dropout_rate=dropout_rate)    
        self.ffn = FeedForward(dim=d_model, dropout_rate=dropout_rate)

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=d_model)

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # sub - layer 1
        norm_x = self.layer_norm_1(x)
        attention_context = self.attention(norm_x, norm_x, norm_x, mask)
        sublayer_1 = self.dropout_1(attention_context) + x

        # sub - layer 2
        norm_sublayer = self.layer_norm_2(sublayer_1)
        ffn_output = self.ffn(norm_sublayer)
        sublayer_2 = self.dropout_2(ffn_output) + sublayer_1

        return sublayer_2
    

class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.local_attention = MultiHeadAttention(d_model=d_model, heads=heads, dropout_rate=dropout_rate)
        self.global_attention = MultiHeadAttention(d_model=d_model, heads=heads, dropout_rate=dropout_rate)
        self.ffn = FeedForward(dim=d_model, dropout_rate=dropout_rate)

        self.layer_norm_1 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=d_model)
        self.layer_norm_3 = nn.LayerNorm(normalized_shape=d_model)

        self.dropout_1 = nn.Dropout(dropout_rate)
        self.dropout_2 = nn.Dropout(dropout_rate)
        self.dropout_3 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, global_context: torch.Tensor, look_ahead_mask: Optional[torch.Tensor] = None, padding_mask: Optional[torch.Tensor] = None):
        # sublayer 1
        norm_x = self.layer_norm_1(x)
        local_attention = self.local_attention(norm_x, norm_x, norm_x, look_ahead_mask)
        local_attention = x + self.dropout_1(local_attention)

        # sublayer 2
        local_norm = self.layer_norm_2(local_attention)
        global_attention = self.global_attention(local_norm, global_context, global_context, padding_mask)
        global_attention = local_norm + self.dropout_2(global_attention)

        # sublayer 3
        global_norm = self.layer_norm_3(global_attention)
        ffn_out = self.ffn(global_norm)
        ffn_out = global_attention + self.dropout_3(ffn_out)

        return ffn_out
