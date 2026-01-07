
import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm

from scripts.pdlm_eval.data_loader import get_eval_data_iterator
from scripts.pdlm_eval.core import load_pdlm_model, generate_single_sample
from scripts.pdlm_eval.metrics import parse_debug_into_blocks, calculate_block_stats, aggregate_metrics
from scripts.pdlm_eval.visualizer import plot_block_trajectory, plot_step_distribution, plot_multi_bucket_convergence
from nanochat.tokenizer import get_tokenizer

def main():
    parser = argparse.ArgumentParser(description="PDLM Evaluation Pipeline")
    parser.add_argument("--model-tag", type=str, default=None, help="Specific model tag to load")
    parser.add_argument("--dataset", type=str, default="duoduoyeah/TinyStories", help="HF Dataset name")
    parser.add_argument("--dataset-config", type=str, default=None, help="HF Dataset config")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--out-dir", type=str, default="eval_results", help="Output directory")
    
    parser.add_argument("--prefix-len", type=int, default=16, help="Length of prompt")
    parser.add_argument("--new-tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--bucket-size", type=int, default=8, help="Bucket size for PDLM")
    
    args = parser.parse_args()
    
    # 1. Load Model
    model, device, autocast_ctx = load_pdlm_model(model_tag=args.model_tag)
    tokenizer = get_tokenizer()
    
    # 2. Prepare Data
    data_iter = get_eval_data_iterator(
        dataset_name=args.dataset,
        dataset_config=args.dataset_config,
        split=args.split,
        max_samples=args.samples,
        prefix_len=args.prefix_len
    )
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    all_raw_blocks = []
    generation_logs = []
    
    print(f"\nStarting Evaluation on {args.samples} samples...")
    pbar = tqdm(total=args.samples)
    
    for i, (prompt_ids, ref_ids, raw_text) in enumerate(data_iter):
        # 3. Run Inference
        # We assume prompt_ids is a list of ints
        new_tokens, block_debug = generate_single_sample(
            model, device, autocast_ctx, 
            prompt_ids, 
            max_new_tokens=args.new_tokens,
            bucket_size=args.bucket_size
        )
        
        # Log text
        prompt_text = tokenizer.decode(prompt_ids)
        response_text = tokenizer.decode(new_tokens)
        generation_logs.append({
            "sample_idx": i,
            "prompt": prompt_text,
            "response": response_text
        })
        
        # 4. Parse Metrics immediately to save memory (optional, but good practice)
        blocks = parse_debug_into_blocks(block_debug)
        all_raw_blocks.extend(blocks)
        
        pbar.update(1)
        
    pbar.close()
    
    if not all_raw_blocks:
        print("No blocks were processed! Check dataset loading or model generation.")
        return
        
    # Save generations
    with open(os.path.join(args.out_dir, "generations.json"), "w") as f:
        json.dump(generation_logs, f, indent=2)

    print("Computing Statistics...")
    # 5. Compute Aggregate Stats
    stats = calculate_block_stats(all_raw_blocks)
    summary = aggregate_metrics(stats)
    
    print("\nEvaluation Summary:")
    print(json.dumps(summary, indent=2))
    
    # 6. Save & Visualize
    # Save summary
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    # Plot Distribution
    plot_step_distribution(stats, args.out_dir)
    
    # Plot Multi-bucket Convergence
    plot_multi_bucket_convergence(stats, args.out_dir)
    
    # Plot Trajectories for the first few blocks (to avoid spamming thousands of images)
    print("Generating trajectory plots for first 5 blocks...")
    for j in range(min(5, len(stats["block_durations"]))):
        plot_block_trajectory(stats, j, os.path.join(args.out_dir, "plots"))

    print(f"\nDone! Results saved to {args.out_dir}")

if __name__ == "__main__":
    main()
