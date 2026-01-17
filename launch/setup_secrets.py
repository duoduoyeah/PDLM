"""
Run this in a Colab cell ONCE to create .env file with secrets.

Usage in Colab:
    %run launch/setup_secrets.py
"""
import os
from google.colab import userdata

env_path = "launch/.env"

print("=== Creating .env file from Colab Secrets ===")

lines = []

# Get HF_TOKEN
try:
    hf_token = userdata.get('HF_TOKEN')
    lines.append(f'export HF_TOKEN="{hf_token}"')
    print(f"OK: HF_TOKEN ({len(hf_token)} chars)")
except Exception as e:
    print(f"ERROR: HF_TOKEN - {e}")
    print("  -> Colab sidebar -> key icon -> Add 'HF_TOKEN'")

# Get WANDB_API_KEY
try:
    wandb_key = userdata.get('WANDB_API_KEY')
    lines.append(f'export WANDB_API_KEY="{wandb_key}"')
    print(f"OK: WANDB_API_KEY ({len(wandb_key)} chars)")
except Exception as e:
    print(f"ERROR: WANDB_API_KEY - {e}")
    print("  -> Colab sidebar -> key icon -> Add 'WANDB_API_KEY'")

# Write .env file
with open(env_path, 'w') as f:
    f.write('\n'.join(lines) + '\n')

print(f"\nCreated: {env_path}")
print("Now run: !bash launch/run_bd3lm_test.sh bd3lm_d4_b4_normal")
