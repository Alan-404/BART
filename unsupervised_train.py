import torch
from torch.utils.data import DataLoader

from processing.processor import BARTProcessor
from module import BARTModule
from dataset import UnsupervisedBARTDataset

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.strategies import DDPStrategy    

from typing import Optional

import fire

def train(
        train_path: str,
        checkpoint: Optional[str] = None,
        saved_checkpoint: str = './checkpoints',
        tokenizer_path: str = "./tokenizer",
        n_layers: int = 6,
        d_model: int = 768,
        heads: int = 12,
        dropout_rate: float = 0.1,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        sep_token: str = "<sep>",
        mask_token: str = "<mask>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        eow_token: str = "</w>",
        num_train: Optional[int] = None,
        batch_size: int = 1,
        num_epochs: int = 1,
        num_workers: int = 1):
    
    processor = BARTProcessor(
        tokenizer_path=tokenizer_path,
        pad_token=pad_token,
        unk_token=unk_token,
        sep_token=sep_token,
        mask_token=mask_token,
        bos_token=bos_token,
        eos_token=eos_token,
        eow_token=eow_token
    )

    if checkpoint is None:
        module = BARTModule(
            token_size=processor.get_token_size(),
            n_layers=n_layers,
            d_model=d_model,
            heads=heads,
            dropout_rate=dropout_rate,
            pad_idx=processor.pad_idx
        )
    else:
        module = BARTModule.load_from_checkpoint(checkpoint, pad_idx=processor.pad_idx)

    def get_batch(batch: torch.Tensor):
        encoder_inputs, encoder_input_lengths = processor(processor.masking(batch), return_lengths=True)
        decoder_inputs, decoder_input_lengths = processor(processor.add_bound(batch), return_lengths=True)
        return encoder_inputs, decoder_inputs, encoder_input_lengths, decoder_input_lengths
    
    dataset = UnsupervisedBARTDataset(train_path, processor=processor, num_examples=num_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=get_batch, num_workers=num_workers)
    
    callbacks = []
    callbacks.append(ModelCheckpoint(dirpath=saved_checkpoint, filename="{epoch}", save_on_train_epoch_end=True, save_last=True))
    
    strategy = 'auto'
    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(process_group_backend='gloo')

    trainer = Trainer(callbacks=callbacks, strategy=strategy, precision='16-mixed', max_epochs=num_epochs)

    trainer.fit(module, train_dataloaders=dataloader, ckpt_path=checkpoint)

if __name__ == '__main__':
    fire.Fire(train)