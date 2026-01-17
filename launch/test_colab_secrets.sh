#!/bin/bash

## Test script to verify Colab secrets are accessible
## Run this first to make sure HF_TOKEN and WANDB_API_KEY are set correctly

echo "=== Testing Colab Secrets ==="

# Test HF_TOKEN
echo "Testing HF_TOKEN..."
HF_TOKEN=$(python -c "from google.colab import userdata; print(userdata.get('HF_TOKEN'))" 2>/dev/null)
if [ -z "${HF_TOKEN}" ] || [ "${HF_TOKEN}" = "None" ]; then
    echo "ERROR: HF_TOKEN not found in Colab secrets"
    echo "  -> Go to Colab sidebar -> Click key icon -> Add 'HF_TOKEN'"
else
    echo "OK: HF_TOKEN found (${#HF_TOKEN} chars)"
fi

# Test WANDB_API_KEY
echo "Testing WANDB_API_KEY..."
WANDB_API_KEY=$(python -c "from google.colab import userdata; print(userdata.get('WANDB_API_KEY'))" 2>/dev/null)
if [ -z "${WANDB_API_KEY}" ] || [ "${WANDB_API_KEY}" = "None" ]; then
    echo "ERROR: WANDB_API_KEY not found in Colab secrets"
    echo "  -> Go to Colab sidebar -> Click key icon -> Add 'WANDB_API_KEY'"
else
    echo "OK: WANDB_API_KEY found (${#WANDB_API_KEY} chars)"
    # Test wandb login
    echo "Testing wandb login..."
    wandb login --relogin "${WANDB_API_KEY}" && echo "OK: wandb login successful" || echo "ERROR: wandb login failed"
fi

echo "=== Done ==="
