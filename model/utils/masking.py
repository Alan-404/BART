import torch
import numpy as np

def generate_mask(lengths: torch.Tensor):
    max_len = torch.max(lengths)
    mask = []
    for length in lengths:
        mask.append(torch.tensor(np.array([1] * int(length.item()) + [0] * int(max_len.item() - length.item()))))
    return torch.stack(mask)

def __generate_look_ahead_mask(length: int) -> torch.Tensor:
    return torch.triu(torch.ones((length, length)), diagonal=1)

def generate_look_ahead_mask(x: torch.Tensor) -> torch.Tensor:
    padding_mask = generate_mask(x)

    look_ahead_mask = __generate_look_ahead_mask(x.size(1)).to(x.device)

    look_ahead_mask = torch.maximum(look_ahead_mask, padding_mask)

    return look_ahead_mask