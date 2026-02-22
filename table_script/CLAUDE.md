# Notes for Agent

- Use `$SCRATCH` for all large file storage (models, datasets, tmp); home dir quota is 50GB.
- Override `--local_dir` in eval scripts to `$SCRATCH/...` instead of `/tmp/`.
- Set `UV_CACHE_DIR=$SCRATCH/.uv_cache` before running uv.
- Do NOT clone the repo in slurm scripts — run directly from `~/PDLM`.
- Results are in `table_script/results/`; they are pushed to `duoduoyeah/eval_results` on HuggingFace via `--push_results`.

## Short evals (~10 min): use srun instead of sbatch
For quick/small evals, prefer an interactive session over submitting a batch job:
```bash
srun --partition=gpu --gres=gpu:a100:1 --mem=50G --time=0:30:00 --pty bash
# then inside the node:
cd ~/PDLM
source launch/.env && export HF_TOKEN
source $SCRATCH/.venv_pdlm/bin/activate
bash table_script/table1_ar.sh
```
Warning: job dies if terminal closes or SSH drops.
