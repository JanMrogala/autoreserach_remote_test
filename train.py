import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import MNISTModel


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST("data", train=True, download=True, transform=transform)
    val_ds = datasets.MNIST("data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, num_workers=4, persistent_workers=True)

    model = MNISTModel(cfg.model)

    logger = WandbLogger(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        log_model=False,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        accelerator="auto",
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader, val_loader)

    # Save code/config as W&B artifacts (Bug fix: was missing)
    if "save_artifacts" in cfg:
        artifact = wandb.Artifact(f"experiment-{wandb.run.id}", type="code")
        for path in cfg.save_artifacts:
            try:
                artifact.add_file(path)
            except Exception as e:
                print(f"Warning: could not save {path}: {e}")
        wandb.log_artifact(artifact)

    # Ensure W&B run is marked complete (Bug fix: was missing)
    wandb.finish()


if __name__ == "__main__":
    main()
