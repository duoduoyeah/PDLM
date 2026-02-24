All slurm job scripts must source setup.sh relative to their own location for uv/venv/wandb setup:
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup.sh"
```
This auto-detects the repo root from the calling script, so each branch/worktree uses its own code.
