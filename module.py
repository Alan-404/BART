from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

import lightning as L

from torchmetrics.text import ROUGEScore, BLEUScore, Perplexity

from model.bart import BART
import statistics
from typing import Optional, List, Union, Tuple

class BARTModule(L.LightningModule):
    def __init__(self, token_size: int, n_layers: int, d_model: int, heads: int, dropout_rate: float = 0.0, pad_idx: Optional[int] = None) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['pad_idx'])

        if pad_idx is None:
            pad_idx = -100

        self.model = BART(
            token_size=token_size,
            n_layers=n_layers,
            d_model=d_model,
            heads=heads,
            dropout_rate=dropout_rate
        )

        self.criterion = BARTCriterion(ignore_index=pad_idx)
        self.metric = BARTMetric(pad_idx=pad_idx)

        self.train_loss = []
    
    # def forward(self, x: torch.Tensor, y: torch.Tensor, x_lengths: Optional[torch.Tensor] = None, y_lengths: Optional[torch.Tensor] = None):
    #     outputs, encoder_outputs = self.model(x, y, x_lengths, y_lengths)
    #     encoder_outputs = self.encoder_linear(encoder_outputs)
    #     return outputs, encoder_outputs
    
    def training_step(self, batch: Tuple[torch.Tensor], _: int):
        x = batch[0]
        observations = batch[1][:, :-1]
        y = batch[1][:, 1:]

        x_lengths = batch[2]
        y_lengths = batch[3] - 1

        outputs = self(x, observations, x_lengths, y_lengths)

        loss = self.criterion(outputs, y)
        self.train_loss.append(loss.item())

        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=3e-5, weight_decay=1e-6, betas=[0.9, 0.98], eps=1e-9)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=1000)
        return [optimizer], [{'scheduler': scheduler, 'interval': "epoch"}]
    
    def on_train_epoch_end(self):
        loss = statistics.mean(self.train_loss)
        print(f"Train Loss: {(loss):.4f}")
        print(f"Current Learning Rate: {self.optimizers().param_groups[0]['lr']}")

        self.log("train_loss", loss, rank_zero_only=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], rank_zero_only=True)
        
        self.train_loss.clear()

class BARTCriterion:
    def __init__(self, ignore_index: int = -100) -> None:
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def __call__(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(outputs.transpose(1, 2), labels)
    
class BARTMetric:
    def __init__(self, pad_idx: Optional[int] = None) -> None:
        self.rouge_1 = ROUGEScore(rouge_keys='rouge1')
        self.rouge_2 = ROUGEScore(rouge_keys='rouge2')
        self.rouge_L = ROUGEScore(rouge_keys='rougeL')

        self.bleu = BLEUScore()

        if pad_idx is not None:
            self.perplexity = Perplexity(ignore_index=pad_idx)

    def rouge_score(self, preds: Union[str, List[str]], labels: Union[str, List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rouge1 = self.rouge_1(preds, labels)
        rouge2 = self.rouge_2(preds, labels)
        rougeL = self.rouge_L(preds, labels)

        return rouge1, rouge2, rougeL
    
    def bleu_score(self, preds: Union[str, List[str]], labels: Union[str, List[str]]) -> torch.Tensor:
        return self.bleu(preds, labels)
    
    def perplexity_score(self, preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert hasattr('perplexity')
        return self.perplexity(preds, labels)