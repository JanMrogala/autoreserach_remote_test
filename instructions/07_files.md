# File Reference

## Files You Edit

| File | Purpose | When to change |
|------|---------|----------------|
| `configs/config.yaml` | Hyperparameters | Every experiment |
| `model.py` | Model architecture | When changing the model |
| `train.py` | Training loop, data loading, logging | When changing training logic |
| `run.sh` | Cluster entrypoint | Rarely (e.g., adding pre-processing steps) |
| `requirements.txt` | Python dependencies | When you need new packages |
| `experiment.def` | Container definition | Rarely (triggers cluster rebuild) |

## Files You Don't Edit

| File | Purpose |
|------|---------|
| `cer` | CLI client (talks to MCP server) |
| `README.md` | Documentation |
| `demo.ipynb` | Demo notebook |

## Directories

| Directory | Purpose | In git? |
|-----------|---------|---------|
| `configs/` | Hydra config files | Yes |
| `workspaces/` | Agent worktrees (one per experiment line) | No (gitignored) |
| `data/` | Downloaded datasets (e.g., MNIST) | No (gitignored) |
| `outputs/` | Hydra output directory | No (gitignored) |
| `wandb/` | Local W&B run cache | No (gitignored) |

## What Gets Sent to the Cluster

When you `./cer submit`, the cluster clones your repo and checks out the exact commit inside a Singularity container. Everything tracked in git at that commit is available. Gitignored files (data, outputs, wandb cache) are not sent — datasets must be downloaded or bind-mounted on the cluster side.

When the job ends, the container's writable layer is discarded automatically. No cleanup needed.

## What Gets Preserved After Reset

When you `./cer workspace reset`, the local worktree and branch are deleted. The only surviving record is:

1. W&B run data (metrics, config, summaries) — always preserved
2. W&B artifacts (files listed in `save_artifacts`) — preserved if you added artifact saving to `train.py`
3. Git history on `main` — if you merged your branch before resetting
