#%%
from processing.processor import BARTProcessor
# %%
processor = BARTProcessor(
    tokenizer_path='./tokenizer',
    return_tokens=False
)
# %%
processor.text2token('Tổng đài viên: xin chào anh\nKhách hàng: chào em', masking=True)
# %%

# %%
seqs = [
    'xin chào tôi là trí',
    'Tổng đài viên: xin chào anh'
]
# %%
processor.text2token('Tổng đài viên: xin chào anh\nKhách hàng: chào em')
# %%
