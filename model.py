import torch
import torch.nn as nn
import pytorch_lightning as pl


class MNISTModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        layers = []
        in_dim = 28 * 28
        for _ in range(cfg.num_layers):
            layers.extend([
                nn.Linear(in_dim, cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
            ])
            in_dim = cfg.hidden_dim
        layers.append(nn.Linear(in_dim, 10))

        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
