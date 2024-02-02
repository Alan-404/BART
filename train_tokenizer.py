import fire
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from typing import Optional

def main(saved_path: str,
         vocab_size: Optional[int] = None,
         pad_token: str = "<pad>",
         unk_token: str = "<unk>",
         sep_token: str = "<sep>",
         mask_token: str = "<mask>",
         bos_token: str = "<bos>",
         eos_token: str = "<eos>",
         eow_token: str = "</w>"):
    
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(vocab_size=vocab_size ,special_tokens=[pad_token, unk_token, sep_token, mask_token, bos_token, eos_token, eow_token], end_of_word_suffix=eow_token)
    tokenizer.train(files=['./datasets/corpus-full.txt'], trainer=trainer)

    tokenizer.model.save(saved_path)

if __name__ == '__main__':
    fire.Fire(main)