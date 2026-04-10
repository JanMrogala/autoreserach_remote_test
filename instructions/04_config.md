# Experiment Configuration

Experiments are configured via `configs/config.yaml` using Hydra. The full config is automatically logged to W&B on every run.

## Config Structure

```yaml
model:
  hidden_dim: 128       # Hidden layer size
  num_layers: 2         # Number of layers
  dropout: 0.1          # Dropout rate
  lr: 1e-3              # Learning rate

training:
  max_epochs: 10        # Training epochs
  batch_size: 64        # Batch size

wandb:
  project: ${oc.env:WANDB_PROJECT,my-project}  # Set by CER, don't change
  entity: ${oc.env:WANDB_ENTITY,null}           # Set by CER, don't change

save_artifacts:         # Files preserved as W&B artifacts after training
  - "configs/config.yaml"
  - "model.py"
  - "train.py"
```

## What to Change

- **model section** — Architecture hyperparameters. These are passed to `model.py` as `cfg.model`.
- **training section** — Training loop parameters. Used in `train.py` as `cfg.training`.
- **save_artifacts** — Add any new files you create that should be preserved (e.g., `utils.py`, `data_loader.py`).

## What NOT to Change

- **wandb section** — These use environment variable interpolation (`${oc.env:...}`). CER injects the correct values at runtime.

## Adding New Config Fields

You can add any fields you need. They're accessible in code via the `cfg` object:

```yaml
model:
  hidden_dim: 256
  activation: "gelu"     # new field

training:
  warmup_steps: 100      # new field
```

```python
# In model.py or train.py:
act = cfg.model.activation  # "gelu"
warmup = cfg.training.warmup_steps  # 100
```

## Numeric Values

YAML scientific notation like `lr: 1e-3` can sometimes be parsed as strings. If you see type errors in the optimizer, cast to float:

```python
torch.optim.Adam(self.parameters(), lr=float(cfg.lr))
```
