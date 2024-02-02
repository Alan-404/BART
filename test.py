#%%
from torchtext.transforms import CharBPETokenizer
# %%
tokenzier = CharBPETokenizer(
    bpe_encoder_path='./tokenizer/merges.txt',
    bpe_merges_path='./tokenizer/vocab.json',
    unk_token="<unk>",
    suffix="</w>",
    special_tokens=["<pad>", "<unk>", "<sep>", "<mask>", "<bos>", "<eos>", "</w>"]
)
# %%
