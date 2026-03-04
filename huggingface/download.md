# HF Download

```bash
# load HF_TOKEN from .env
source launch/.env
export HF_TOKEN

# one-time login
uvx hf auth login

# download from private repo
uvx hf download duoduoyeah/eval_results \
    --repo-type dataset \
    --include "bd3lm/*/seq*_threshold/*" \
    --local-dir ./table_script/results/table6_bd3lm
```
