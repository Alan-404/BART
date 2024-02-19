#%%
from processing.processor import BARTProcessor
import torch
from model.bart import BART
# %%
processor = BARTProcessor('./tokenizer')
# %%
model = BART(
    token_size=processor.get_token_size(),
    n_layers=6,
    d_model=768,
    heads=12,
    dropout_rate=0.1
)
# %%
diag = "Tổng đài viên: không phải là số điện hạ chị nguyễn thị cẩm nhung đúng không\nKhách hàng: đúng rồi\nTổng đài viên: chị cho em hỏi thăm đường truyền mạng của fpt chị mình sử dụng có vấn đề gì cần bên em hỗ trợ không chị\nKhách hàng: còn chậm lắm chị ơi\nTổng đài viên: trơn hả chị em thấy nhà mình đã tắt bị chị không biết là mình đi vắng hay sao vậy chị hả\nKhách hàng: hả\nTổng đài viên: em thấy là thiết bị nhà mình đang bị tắt chị không biết là mình đi vắng hay sao\nKhách hàng: sao chị ý chị nói gì\nTổng đài viên: em thấy là cái thiết bị wifi nhà mình đang rút cái modem ra là không biết là mình đi vắng hay sao\nKhách hàng: để em liên hệ với người wifi xuống đây em không để ý giờ em đang bận xíu chị\nTổng đài viên: là ở một trăm mười tám thật đúng không chị để em nhân viên luôn\nKhách hàng: đúng rồi đúng rồi\nTổng đài viên: để em báo nhân viên luôn chị\nKhách hàng: ô kê ô kê em đang bận thu\nTổng đài viên: cảm ơn chị"
summary = '<agent> đã kiểm tra vấn đề kết nối mạng của khách hàng và phát hiện rằng thiết bị wifi đã bị tắt, sau đó thông báo và đề xuất liên hệ với nhân viên để khắc phục tình trạng. Khách hàng thông báo về tình trạng kết nối mạng chậm, sau đó không nhận ra rằng thiết bị wifi đã bị tắt. Khách hàng đồng ý để tổng đài viên thông báo vấn đề cho nhân viên và cho biết đang bận.'
# %%
x = processor.text2token(diag, masking=True)
# %%
y = processor.text2token(summary)
# %%
processor.token2text(y)
# %%
processor.vocab[6884]
# %%
obs = y[:-1]
y = y[1:]
# %%
x = x.unsqueeze(0)
# %%
obs = obs.unsqueeze(0)
y = y.unsqueeze(0)
# %%
out = model(x, obs)
# %%
out[0].shape
# %%
out[1].shape
# %%
x.shape
# %%
y.shape
# %%
import torch
from model.utils.masking import generate_look_ahead_mask, generate_mask
# %%
x_lengths = torch.tensor([45, 76, 12])
y_lengths = torch.tensor([45, 50, 12])
# %%
padding_mask = generate_mask(x_lengths)
#%%
padding_mask = (padding_mask == 0).unsqueeze(1).unsqueeze(1)
# %%
look_mask = generate_look_ahead_mask(y_lengths)
# %%
a = torch.rand((1, 12, 76, 64))
b = torch.rand((1, 12, 50, 64))
# %%
c = torch.matmul(b, a.transpose(-1, -2))
# %%
c.shape
# %%
c.masked_fill(padding_mask, float('-inf')).shape
# %%
import io
# %%
data = io.open('./datasets/corpus-full.txt', encoding='utf-8').read().strip().split('\n')
# %%
