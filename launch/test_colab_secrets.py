"""
Run this in a Colab cell to set up secrets before running bash scripts.

Usage in Colab:
    %run launch/test_colab_secrets.py
    !bash launch/run_bd3lm_test.sh bd3lm_d4_b4_normal
"""
import os
from google.colab import userdata

print("=== Setting up Colab Secrets ===")

# Get and export HF_TOKEN
try:
    hf_token = userdata.get('HF_TOKEN')
    os.environ['HF_TOKEN'] = hf_token
    print(f"OK: HF_TOKEN set ({len(hf_token)} chars)")
except Exception as e:
    print(f"ERROR: HF_TOKEN - {e}")
    print("  -> Go to Colab sidebar -> Click key icon -> Add 'HF_TOKEN'")

# Get and export WANDB_API_KEY
try:
    wandb_key = userdata.get('WANDB_API_KEY')
    os.environ['WANDB_API_KEY'] = wandb_key
    print(f"OK: WANDB_API_KEY set ({len(wandb_key)} chars)")

    # Login to wandb
    import wandb
    wandb.login(key=wandb_key, relogin=True)
    print("OK: wandb login successful")
except Exception as e:
    print(f"ERROR: WANDB_API_KEY - {e}")
    print("  -> Go to Colab sidebar -> Click key icon -> Add 'WANDB_API_KEY'")

print("=== Done ===")
print("\nNow run: !bash launch/run_bd3lm_test.sh bd3lm_d4_b4_normal")
