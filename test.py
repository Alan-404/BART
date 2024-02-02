#%%
import torch
from model.bart import BART
# %%
model = BART(
    token_size=50000,
    n_layers=6,
    d_model=768,
    heads=12,
    dropout_rate=0.1
)
# %%
a = torch.tensor([[2,3,4,5,6,7,8,9]])
b = torch.tensor([[2,3,4,5,6,7,8,9]])
# %%
out = model(a, b)
# %%
out.shape
# %%
