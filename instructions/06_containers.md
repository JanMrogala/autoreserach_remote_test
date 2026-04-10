# Container Setup

Two containers are involved in the CER pipeline. You run inside one locally, and your experiments run inside the other on the cluster.

## Local Container (you are here)

Built from `experiment.def` using Apptainer. This is where you (the agent) run. It has:

- Python 3 + PyTorch (from NVIDIA base image)
- Standard ML libraries: wandb, pytorch-lightning, hydra-core, scipy, matplotlib
- MCP client (for `./cer` commands)
- Git (for version control)
- No SSH, no internet access (only localhost:8000 for MCP)

The workspace is bind-mounted from the host, so your file edits persist outside the container.

## Cluster Container

The same `experiment.def` is used to build a Singularity container on the LUMI cluster. The container must be pre-built on the cluster before submitting jobs.

On each job, CER launches Singularity with `--writable-tmpfs`. Inside the container:
1. Your repo is cloned into a temporary directory
2. The exact commit you submitted is checked out
3. `pip install -r requirements.txt` installs any extra deps
4. Your experiment entrypoint runs

When the job ends (success or failure), the container's writable layer is discarded. No cleanup needed — nothing persists on the cluster filesystem except logs.

## When to Modify `experiment.def`

Only modify it when you need system-level packages or new pip dependencies that aren't in `requirements.txt`. For example:

- Need a new C library (e.g., `libffi-dev`) -> edit `experiment.def`
- Need a new Python package -> prefer adding to `requirements.txt` first (installed at container start)
- Only add to `experiment.def` if the package is large or slow to install (it gets baked into the image)

## Container Definition Format

```
Bootstrap: docker
From: nvcr.io/nvidia/pytorch:24.07-py3

%post
    apt-get update && apt-get install -y --no-install-recommends \
        package-name \
        && rm -rf /var/lib/apt/lists/*

    pip install --no-cache-dir package-name

%environment
    export LC_ALL=C.UTF-8
    export LANG=C.UTF-8
```

## Important

- Changing `experiment.def` requires rebuilding the container on the cluster (human action, takes 10+ minutes)
- The `requirements.txt` route is faster — packages install at job start without rebuilding
- Keep `experiment.def` changes infrequent and batch them
