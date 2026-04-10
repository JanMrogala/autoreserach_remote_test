# Experiment Workflow

## Standard Loop

```
1. Create workspace (once)
2. Form hypothesis
3. Edit code/config
4. Commit
5. Submit
6. Wait for completion
7. Analyze results
8. Go to 2 (or reset workspace for a new direction)
```

## Step-by-Step

### 1. Setup (once per experiment line)
```bash
./cer workspace create my-experiment
cd workspaces/my-experiment
```

### 2. Make changes

Edit any of these files:
- `configs/config.yaml` — change hyperparameters (learning rate, layers, batch size, etc.)
- `model.py` — change model architecture
- `train.py` — change training logic, add metrics, modify logging

### 3. Commit and submit
```bash
git add -A
git commit -m "experiment: what you changed and why"
./cer submit $(git rev-parse HEAD)
```
Note the job ID from the output.

### 4. Monitor
```bash
./cer status <job_id>
```
Repeat until status is `COMPLETED` or `FAILED`. SLURM statuses:
- `PENDING` — waiting in queue
- `RUNNING` — executing on GPU
- `COMPLETED` — finished successfully
- `FAILED` — crashed (check results/logs for errors)
- `CANCELLED` — manually cancelled

### 5. Analyze results
```bash
./cer results <job_id> --history --key train_loss --key val_loss --key val_acc
```
Parse the JSON output to decide your next move.

### 6. Iterate or reset

**Iterate** — make more changes in the same workspace and submit again.

**Reset** — if you want to start from a clean `main`:
```bash
cd ../..
./cer workspace reset my-experiment
cd workspaces/my-experiment
```

## Tips

- Each commit = one experiment. Keep changes focused so you know what caused each result.
- Write descriptive commit messages — they're the only record of what you tried.
- Don't change multiple things at once unless you have a good reason. Isolate variables.
- Check `./cer list` to see all past experiments and their statuses.
- If a job fails, check `./cer results <job_id>` — it may have partial logs.
- The workspace limit is 10. Clean up old workspaces with `./cer workspace reset`.
