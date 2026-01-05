# PDLM Evaluation Pipeline

This folder contains a comprehensive evaluation pipeline for the **Parallel Denoising Language Model (PDLM)**. It assesses the model's generation efficiency, stability, and convergence characteristics by running it against standard datasets (like `TinyStories` or `WikiText`).

## Quick Start

Run the evaluation from the project root directory:

```bash
# Basic run with default settings (TinyStories, 10 samples)
python -m scripts.pdlm_eval.run_eval

# Customize the run
python -m scripts.pdlm_eval.run_eval \
    --dataset duoduoyeah/TinyStories \
    --split validation \
    --samples 20 \
    --prefix-len 16 \
    --new-tokens 256 \
    --out-dir my_eval_results
```

## Key Metrics & Outputs

The script generates a `summary.json` and several plots in the output directory:

1.  **`avg_steps_per_block`**: Average number of denoising steps required to generate a bucket (block) of tokens.
2.  **`avg_bucket_convergence`**: The average step number where *all* tokens in a bucket stabilize and match the final output.
3.  **Plots**:
    *   **`steps_distribution.png`**: Histogram of steps needed per block.
    *   **`multi_bucket_convergence.png`**: Shows how many tokens "lock in" (converge) at each step for different buckets.
    *   **`plots/block_X_traj.png`**: Detailed probability trajectory for individual tokens within a block. Solid lines indicate the final chosen token; dashed lines indicate temporary guesses.

## Modules

*   **`run_eval.py`**: Main entry point and orchestrator.
*   **`data_loader.py`**: Handles dataset streaming and tokenization (splits text into Prompt vs. Reference).
*   **`core.py`**: Wraps the PDLM model interaction, ensuring correct loading and debug data extraction.
*   **`metrics.py`**: Calculates convergence steps and stability stats from the raw debug stream.
*   **`visualizer.py`**: Generates the matplotlib graphs.
