import os
import torch
from typing import List
import numpy as np
from torchtext.transforms import CharBPETokenizer
import re
import random

class BARTProcessor:
    def __init__(self, 
                 tokenizer_path: str, 
                 pad_token: str = "<pad>", 
                 unk_token: str = "<unk>", 
                 sep_token: str = "<sep>", 
                 mask_token: str = "<mask>", 
                 bos_token: str = "<bos>", 
                 eos_token: str = "<eos>", 
                 eow_token: str = "</w>",
                 return_tokens: bool = False,
                 puncs: List[str] = r"([:./,?!@#$%^&=`~*\(\)\[\]\"\'\-\\])") -> None:
        assert os.path.exists(tokenizer_path) and os.path.exists(f"{tokenizer_path}/vocab.json") and os.path.exists(f"{tokenizer_path}/merges.txt")

        self.tokenizer = CharBPETokenizer(
            bpe_encoder_path=f"{tokenizer_path}/vocab.json",
            bpe_merges_path=f"{tokenizer_path}/merges.txt",
            unk_token=unk_token,
            suffix=eow_token,
            return_tokens=return_tokens,
            special_tokens=[pad_token, unk_token, sep_token, mask_token, bos_token, eos_token, eow_token]
        )

        self.pad_idx = self.tokenizer._encoder[pad_token]
        self.unk_idx = self.tokenizer._encoder[unk_token]
        self.sep_idx = self.tokenizer._encoder[sep_token]
        self.mask_idx = self.tokenizer._encoder[mask_token]
        self.bos_idx = self.tokenizer._encoder[bos_token]
        self.eos_idx = self.tokenizer._encoder[eos_token]
        self.eow_idx = self.tokenizer._encoder[eow_token]

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.eow_token = eow_token

        self.puncs = puncs

    def get_token_size(self) -> int:
        return len(self.tokenizer._encoder)

    def clean(self, seq: str):
        seq = re.sub(self.puncs, r" \1 ", seq)
        seq = re.sub("\n", f" {self.sep_token} ", seq)
        seq = re.sub("\s\s+", " ", seq)
        seq = seq.strip()
        seq = seq.lower()
        return seq

    def text2token(self, sentence: str, add_token: bool = False, masking: bool = False):
        sentence = self.clean(sentence)
        if add_token:
            sentence = f"{self.bos_token} {sentence} {self.eos_token}"
        tokens = torch.tensor(self.tokenizer(sentence))

        if masking:
            tokens = self.masking(tokens)

        return tokens
    
    def __call__(self, token_seqs: List[torch.Tensor], return_lengths: bool = False):
        lengths = []
        for item in token_seqs:
            lengths.append(len(item))

        lengths = torch.tensor(lengths)

        max_length = torch.max(lengths)

        padded_tokens = []
        for index, item in enumerate(token_seqs):
            padded_tokens.append(torch.concat(item, torch.tensor(np.array([self.pad_idx] * (max_length - lengths[index]))), dim=-1))

        if return_lengths:
            return torch.stack(padded_tokens), lengths

        return torch.stack(padded_tokens)
    
    def masking(self, tokens: torch.Tensor):
        n_ctx = len(tokens)
        num_mask = n_ctx // 4

        if num_mask == 0:
            return tokens

        masking_indexes = []
        for _ in range(random.randint(0, num_mask)):
            masking_indexes.append(random.randint(0, n_ctx-1))

        for index in masking_indexes:
            tokens[index] = self.mask_idx

        return tokens