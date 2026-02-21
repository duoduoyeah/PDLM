#!/bin/bash
# Quick test: verify HF_TOKEN is valid and has write access to duoduoyeah/eval_results

source launch/.env

python -c "
import os
from huggingface_hub import HfApi

token = os.environ.get('HF_TOKEN')
if not token:
    print('ERROR: HF_TOKEN not set')
    exit(1)

api = HfApi()

# Check token identity
user = api.whoami(token=token)
print(f'Logged in as: {user[\"name\"]}')

# Check write access to eval_results
repo_id = 'duoduoyeah/eval_results'
info = api.repo_info(repo_id=repo_id, repo_type='dataset', token=token)
print(f'Repo: {info.id}')
print(f'Private: {info.private}')

# Check if we can write (look for write in permissions)
perms = getattr(info, 'permissions', None)
if perms:
    print(f'Can write: {getattr(perms, \"canWrite\", \"unknown\")}')
else:
    print('Permissions: not returned (likely owner or public repo)')

print('Token OK')
"
