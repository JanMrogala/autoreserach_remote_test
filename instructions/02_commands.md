# Available Commands

All commands use the `./cer` CLI. Output is JSON unless noted.

## Workspace Management (local, no network)

### Create a workspace
```bash
./cer workspace create <name>
```
Creates a git worktree + branch from `main`. Work inside `workspaces/<name>/`.

### List workspaces
```bash
./cer workspace list
```
Returns JSON array of `{name, path, branch, commit}`.

### Reset a workspace
```bash
./cer workspace reset <name>
```
Deletes the worktree and branch (local + remote), then recreates from latest `main`. Use this to start a completely fresh experiment line.

## Experiment Lifecycle (via MCP server)

### Submit an experiment
```bash
./cer submit <commit_hash>
```
Auto-pushes your current branch, then submits the commit to the cluster. Returns the SLURM job ID.

Typical usage:
```bash
git add -A && git commit -m "experiment: description"
./cer submit $(git rev-parse HEAD)
```

### Check job status
```bash
./cer status <job_id>
```
Returns JSON with `status` (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED), commit info, and W&B URL if available.

### Cancel a job
```bash
./cer cancel <job_id>
```

### List all experiments
```bash
./cer list
```
Returns JSON array of all tracked experiments with their current status.

## Results (via W&B)

### Get summary metrics
```bash
./cer results <job_id>
```
Returns final metrics from the W&B run (e.g., best loss, final accuracy).

### Get full training history
```bash
./cer results <job_id> --history
```
Returns all logged metrics at every step.

### Filter specific metrics
```bash
./cer results <job_id> --history --key train_loss --key val_loss
```
Only returns the specified metric keys. Use this to reduce output size.

## Important Notes

- `submit` requires a committed state — uncommitted changes are not sent to the cluster
- `submit` auto-pushes, so you don't need to `git push` manually
- `status` may show PENDING for a while — the cluster queues jobs
- `results` only works after the job has started logging to W&B
- All commands return JSON for easy parsing
