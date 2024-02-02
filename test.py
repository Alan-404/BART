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
import torchsummary
# %%
torchsummary.summary(model, depth=3)
# %%
