from torch.utils.data import Dataset
from processing.processor import BARTProcessor
from typing import Optional

import pandas as pd

class BARTDataset(Dataset):
    def __init__(self, manifest_path: str, processor: BARTProcessor, num_examples: Optional[int] = None) -> None:
        super().__init__()
        self.prompts = pd.read_csv(manifest_path, sep="\t")

        if num_examples is not None:
            self.prompts = self.prompts[:num_examples]

        self.processor = processor

    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, index: int):
        index_df = self.prompts.iloc[index]

        dialogue = index_df['dialogue']
        summary = index_df['summary']

        return self.processor.text2token(dialogue, masking=True), self.processor.text2token(summary, bos_token=True, eos_token=True)