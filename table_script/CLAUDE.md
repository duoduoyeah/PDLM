# Notes for Agent

- Use `$SCRATCH` for all large file storage (models, datasets, tmp); home dir is only 20GB.
- Override `--local_dir` in eval scripts to `$SCRATCH/...` instead of `/tmp/`.
- Set `UV_CACHE_DIR=$SCRATCH/.uv_cache` before running uv.
- Clone the repo itself to `$SCRATCH`, not home.
- Results are in `table_script/results/`; copy them out of `$SCRATCH` before job ends (scratch may be auto-cleaned).
