# WandB Interaction

entity: `lizhicuocuocuo-university-of-california-riverside`, project: `nanochat`

## What we can do
1. **List runs** — list all runs in a group with name/state/id
2. **Run config** — get hyperparameters for a run (`run.config`)
3. **Final metrics** — get final summary stats (`run.summary`)
4. **Loss curves (sampled)** — download ~500pt history (`run.history()`)
5. **Loss curves (full)** — download all steps unsampled (`run.scan_history()`)
6. **Compare runs** — fetch summary of all group runs into one DataFrame
7. **Download files** — download saved artifacts/files (`run.files()`)
8. **Update config** — patch run config post-hoc (`run.config[k]=v; run.update()`)
