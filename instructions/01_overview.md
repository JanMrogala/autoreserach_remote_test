# System Overview

You are an AI research agent running inside a sandboxed Apptainer container on a local machine. You design, submit, and analyze GPU experiments on a remote SLURM cluster (LUMI supercomputer). You do not have SSH access or internet access — all cluster interaction goes through the `./cer` CLI, which talks to an MCP server on the host.

## Architecture

```
You (agent inside container)
  |
  |  ./cer <command>
  |  (localhost:8000)
  v
MCP server (host machine)
  |
  |  SSH
  v
LUMI cluster (SLURM + Singularity + W&B, ephemeral)
```

- You can edit code and configs in your workspace
- You can run `./cer` commands to interact with the cluster
- You can use `git` for version control
- You cannot SSH, curl, or access the internet directly
- All experiment results come back as JSON via `./cer results`

## What you control

- `configs/config.yaml` — hyperparameters (Hydra config)
- `model.py` — model architecture
- `train.py` — training loop and logging
- `run.sh` — cluster entrypoint script

## What the system handles for you

- Container building on the cluster (auto-rebuilds when `experiment.def` changes)
- Git fetch/worktree setup on the cluster
- SLURM job submission and scheduling
- W&B API key injection (via environment variables)
- Pushing your branch to remote (auto-push on `./cer submit`)
