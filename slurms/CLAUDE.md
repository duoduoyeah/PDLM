Slurm scripts must set `REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"` then `source "$REPO_ROOT/slurms/setup.sh"` so each branch uses its own code.
