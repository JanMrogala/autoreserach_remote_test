import torch
import torch.nn as nn
import pytorch_lightning as pl


class MNISTModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg

        arch = getattr(cfg, "arch", "mlp")

        if arch == "cnn":
            self.net = self._build_cnn(cfg)
        else:
            self.net = self._build_mlp(cfg)

        self.loss_fn = nn.CrossEntropyLoss()

    def _build_mlp(self, cfg):
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
        return nn.Sequential(*layers)

    def _build_cnn(self, cfg):
        channels = getattr(cfg, "channels", 32)
        return nn.Sequential(
            # Conv block 1: 1x28x28 -> channels x 14x14
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(cfg.dropout),
            # Conv block 2: channels x 14x14 -> 2*channels x 7x7
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(cfg.dropout),
            # Classifier
            nn.Flatten(),
            nn.Linear(channels * 2 * 7 * 7, cfg.hidden_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 10),
        )

    def forward(self, x):
        arch = getattr(self.cfg, "arch", "mlp")
        if arch == "cnn":
            # CNN expects [B, 1, 28, 28]
            if x.dim() == 3:
                x = x.unsqueeze(1)
            return self.net(x)
        else:
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
        opt_name = getattr(self.cfg, "optimizer", "adam")
        weight_decay = float(getattr(self.cfg, "weight_decay", 0.0))
        lr = float(self.cfg.lr)

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        scheduler_name = getattr(self.cfg, "scheduler", None)
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.trainer.max_epochs,
            )
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

        return optimizer
