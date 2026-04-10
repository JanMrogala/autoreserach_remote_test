# Agent Test Prompt

Copy and paste everything below the line as a prompt for the agent.

---

You are a research agent testing the CER (Cluster Experiment Runner) pipeline. Your goal is to run 2-3 experiment variations on a remote SLURM cluster, verify the full pipeline works end-to-end, and write a detailed report of what worked and what didn't.

## Setup

You are running inside an Apptainer container. You interact with the cluster exclusively through the `./cer` CLI. Read the `instructions/` directory first to understand the available commands, workflow, and file structure.

## Task

### Phase 1: Orientation

1. Read all files in `instructions/` to understand the system.
2. Read `configs/config.yaml`, `model.py`, `train.py`, and `run.sh` to understand the current experiment.
3. Run `./cer list` to check for any existing experiments.

### Phase 2: Workspace Setup

1. Create a workspace: `./cer workspace create test-agent`
2. `cd workspaces/test-agent`
3. Verify the workspace has all expected files.

### Phase 3: Baseline Experiment

1. Do NOT change any code. Commit the current state as the baseline.
2. Submit: `git add -A && git commit -m "baseline: default config" && ./cer submit $(git rev-parse HEAD)`
3. Note the job ID.
4. Poll `./cer status <job_id>` every 60 seconds until the job completes or fails.
5. Once done, fetch results: `./cer results <job_id> --history --key train_loss --key val_loss --key val_acc --key train_acc`
6. Record the final metrics.

### Phase 4: Variation Experiments

Run 2 more experiments with different hyperparameters. For each:

1. Edit `configs/config.yaml` to change hyperparameters. Suggested variations:
   - **Variation 1 — Wider + lower LR**: `hidden_dim: 256`, `num_layers: 3`, `lr: 5e-4`, `batch_size: 128`
   - **Variation 2 — Aggressive training**: `hidden_dim: 512`, `dropout: 0.2`, `lr: 3e-3`, `max_epochs: 10`, `batch_size: 32`
2. Commit with a descriptive message explaining the hypothesis.
3. Submit via `./cer submit $(git rev-parse HEAD)`
4. Wait for completion, fetch results.
5. Compare against baseline.

You may adjust the variations based on baseline results. If the baseline fails, diagnose the issue before proceeding.

### Phase 5: Report

After all experiments finish (or fail), write a file called `TEST_REPORT.md` in the workspace root with this structure:

```markdown
# CER Pipeline Test Report

**Date:** YYYY-MM-DD
**Agent:** <your name/model>
**Workspace:** test-agent

## Summary

<1-2 sentence overall verdict: did the pipeline work end-to-end?>

## Pipeline Steps Tested

| Step | Command | Result | Notes |
|------|---------|--------|-------|
| Read instructions | cat instructions/*.md | OK/FAIL | |
| Create workspace | ./cer workspace create | OK/FAIL | |
| List experiments | ./cer list | OK/FAIL | |
| Submit baseline | ./cer submit | OK/FAIL | Job ID: |
| Status polling | ./cer status | OK/FAIL | |
| Fetch results | ./cer results | OK/FAIL | |
| Submit variation 1 | ./cer submit | OK/FAIL | Job ID: |
| Submit variation 2 | ./cer submit | OK/FAIL | Job ID: |
| Workspace reset | ./cer workspace reset | OK/FAIL | |

## Experiment Results

| Experiment | Commit | Job ID | Status | Final train_loss | Final val_loss | Final val_acc |
|-----------|--------|--------|--------|-----------------|---------------|---------------|
| Baseline | | | | | | |
| Variation 1 | | | | | | |
| Variation 2 | | | | | | |

## Bugs and Errors

For each issue found, document:
- **Severity**: Critical / High / Medium / Low
- **Step**: Which pipeline step failed
- **Command**: Exact command that was run
- **Expected**: What should have happened
- **Actual**: What actually happened (include full error output)
- **Workaround**: If you found one

## Observations

- Any unexpected behavior, slow steps, confusing output, or missing documentation
- Suggestions for improvement
- What worked well

## Raw Logs

<Paste key command outputs here for debugging>
```

Commit the report to the workspace branch before finishing.

## Rules

- Do not modify `cer`, `experiment.def`, or files in `instructions/`.
- If a submit fails, diagnose why before retrying. Do not blindly retry.
- If polling status, wait at least 60 seconds between checks to avoid spamming.
- Record exact commands and their output — the report is useless without specifics.
- If you get stuck on a blocker that prevents all progress, still write the report documenting what failed and why.
