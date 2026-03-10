#!/bin/bash
## Upload eval results to HuggingFace (single commit).
## Reusable by any eval script — call once after all evals finish.
##
## NOTE: --result_dirs must point to paths that still exist when this script runs.
##   - Scratch dirs (/scratch/sli588/<jobid>/) are cleaned up when srun exits,
##     so this script CANNOT batch-upload results from multiple finished sruns.
##   - To use this for batch upload after parallel sruns, either:
##     (a) save eval JSONs to a persistent path (home dir or repo), or
##     (b) rclone results to Google Drive, then upload from there.
##   - For single srun: call this inside the srun (via --push_results) before it exits.
##
## Usage:
##   bash slurms/hf_upload.sh \
##       --repo_prefix=mask_pdlm \
##       --run_folder=seq3200_threshold \
##       --result_dirs="/scratch/.../model_a /scratch/.../model_b"
##
##   # Custom repo:
##   bash slurms/hf_upload.sh \
##       --repo_prefix=bd3lm \
##       --run_folder=seq3200_threshold \
##       --repo_id=duoduoyeah/eval_results \
##       --result_dirs=/scratch/.../bd3lm_model

set -e

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${REPO_ROOT}/slurms/setup.sh"

# ============================================================
# Default values
# ============================================================
REPO_PREFIX=""
RUN_FOLDER=""
RESULT_DIRS=""
REPO_ID="duoduoyeah/eval_results"
EVAL_JSON_NAME="eval_threshold.json"

# Parse arguments
for arg in "$@"; do
    case $arg in
        --repo_prefix=*)
            REPO_PREFIX="${arg#*=}"
            ;;
        --run_folder=*)
            RUN_FOLDER="${arg#*=}"
            ;;
        --result_dirs=*)
            RESULT_DIRS="${arg#*=}"
            ;;
        --repo_id=*)
            REPO_ID="${arg#*=}"
            ;;
        --eval_json=*)
            EVAL_JSON_NAME="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: bash slurms/hf_upload.sh \\"
            echo "    --repo_prefix=mask_pdlm \\"
            echo "    --run_folder=seq3200_threshold \\"
            echo "    --result_dirs=\"/path/model_a /path/model_b\" \\"
            echo "    [--repo_id=duoduoyeah/eval_results]"
            exit 1
            ;;
    esac
done

if [ -z "${REPO_PREFIX}" ] || [ -z "${RUN_FOLDER}" ] || [ -z "${RESULT_DIRS}" ]; then
    echo "Error: --repo_prefix, --run_folder, and --result_dirs are required"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "HuggingFace Upload"
echo "============================================================"
echo "Repo:        ${REPO_ID}"
echo "Prefix:      ${REPO_PREFIX}"
echo "Run folder:  ${RUN_FOLDER}"
echo "Timestamp:   ${TIMESTAMP}"
echo "============================================================"

python -c "
import json, os, tempfile, shutil
from huggingface_hub import upload_folder

repo_prefix    = '${REPO_PREFIX}'
run_folder     = '${RUN_FOLDER}'
repo_id        = '${REPO_ID}'
timestamp      = '${TIMESTAMP}'
eval_json_name = '${EVAL_JSON_NAME}'
result_dirs    = '''${RESULT_DIRS}'''.split()

eval_json_stem = eval_json_name.replace('.json', '')

tmp = tempfile.mkdtemp()
staged = 0
try:
    for result_dir in result_dirs:
        model_name = os.path.basename(result_dir)
        src_json   = os.path.join(result_dir, eval_json_name)

        if not os.path.exists(src_json):
            print(f'[skip] {model_name}: {eval_json_name} not found')
            continue

        dest_dir = os.path.join(tmp, repo_prefix, model_name, run_folder)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.copy(src_json, f'{dest_dir}/{eval_json_stem}_{timestamp}.json')

        args = {
            'model':     model_name,
            'timestamp': timestamp,
        }
        with open(f'{dest_dir}/args_{timestamp}.json', 'w') as f:
            json.dump(args, f, indent=2)

        print(f'[staged] {model_name}')
        staged += 1

    if staged == 0:
        print('Nothing to upload.')
    else:
        upload_folder(
            folder_path=tmp,
            path_in_repo='',
            repo_id=repo_id,
            repo_type='dataset',
            token=os.environ.get('HF_TOKEN'),
        )
        print(f'Uploaded {staged} model(s) to {repo_id}/{repo_prefix}/')
finally:
    shutil.rmtree(tmp)
"
