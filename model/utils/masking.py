import torch

def generate_mask(lengths: torch.Tensor, return_max_length: bool = False):
    max_len = torch.max(lengths)
    mask = []
    for length in lengths:
        mask.append(torch.tensor([1] * int(length.item()) + [0] * int(max_len.item() - length.item())))
    mask = torch.stack(mask).to(lengths.device)

    if return_max_length:
        return mask, max_len
    
    return mask

def get_lower_triangular(length: int) -> torch.Tensor:
    return torch.tril(torch.ones((length, length)), diagonal=0)

def generate_look_ahead_mask(lengths: torch.Tensor) -> torch.Tensor:
    padding_mask, max_length = generate_mask(lengths, return_max_length=True)
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(1)

    triangular = get_lower_triangular(max_length).to(lengths.device)

    look_ahead_mask = torch.minimum(triangular, padding_mask)

    return look_ahead_mask