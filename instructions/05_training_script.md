# Training Script Requirements

The training script (`train.py`) must integrate with Hydra for config and W&B for logging. Without these, CER cannot track your experiments.

## Required Structure

```python
import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # 1. Initialize W&B with full config
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # 2. Your training loop — log metrics with wandb.log()
    for epoch in range(cfg.training.max_epochs):
        train_loss, val_loss, val_acc = train_one_epoch(...)
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

    # 3. Save artifacts (preserves files after workspace cleanup)
    if "save_artifacts" in cfg:
        artifact = wandb.Artifact(f"experiment-{wandb.run.id}", type="code")
        for path in cfg.save_artifacts:
            try:
                artifact.add_file(path)
            except Exception as e:
                print(f"Warning: could not save {path}: {e}")
        wandb.log_artifact(artifact)

    # 4. Always call finish
    wandb.finish()

if __name__ == "__main__":
    main()
```

## Metric Naming

- Use consistent names across experiments so you can compare results
- The metric keys you log are what `./cer results --key <name>` filters on
- Common pattern: `train_loss`, `val_loss`, `train_acc`, `val_acc`

## PyTorch Lightning

If using Lightning, use `WandbLogger` instead of raw `wandb.init()`:

```python
from pytorch_lightning.loggers import WandbLogger

logger = WandbLogger(
    project=cfg.wandb.project,
    entity=cfg.wandb.entity,
    config=OmegaConf.to_container(cfg, resolve=True),
)

trainer = pl.Trainer(logger=logger, ...)
trainer.fit(model, train_loader, val_loader)
```

Lightning auto-logs metrics from `self.log()` calls in your model. After training, add the artifact saving block and `wandb.finish()` manually.

## Environment Variables

CER sets these automatically on the cluster — you don't need to set them:

| Variable | Purpose |
|----------|---------|
| `WANDB_PROJECT` | W&B project name |
| `WANDB_RUN_NAME` | `cer-<commit_short>` |
| `WANDB_TAGS` | Commit hash (how `./cer results` finds your run) |
| `WANDB_API_KEY` | Authentication |
| `CER_COMMIT` | Full commit hash |
