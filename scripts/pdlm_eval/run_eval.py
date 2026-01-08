
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
    
    parser.add_argument("--base-models-dir", type=str, default=None, help="Directory containing multiple model folders to evaluate")
    
    parser.add_argument("--prefix-len", type=int, default=16, help="Length of prompt")
    parser.add_argument("--new-tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--bucket-size", type=int, default=None, help="Bucket size for PDLM (None to use model config)")
    
    args = parser.parse_args()

    if args.base_models_dir:
        # Batch Mode
        base_path = os.path.expanduser(args.base_models_dir)
        if not os.path.exists(base_path):
            print(f"Error: Base models dir {base_path} does not exist.")
            return

        model_folders = sorted([
            f for f in os.listdir(base_path) 
            if os.path.isdir(os.path.join(base_path, f)) and not f.startswith(".")
        ])
        print(f"Found {len(model_folders)} models in {base_path}")
        
        original_out_dir = args.out_dir
        
        for model_folder in model_folders:
            print(f"\n{'='*50}")
            print(f"Evaluating Model: {model_folder}")
            print(f"{'='*50}")
            
            # Set environment variable for this run
            full_model_path = os.path.join(base_path, model_folder)
            os.environ["NANOCHAT_BASE_DIR"] = full_model_path
            
            # Create subfolder for results
            sub_out_dir = os.path.join(original_out_dir, model_folder)
            
            try:
                run_evaluation_for_model(args, sub_out_dir)
            except Exception as e:
                print(f"Failed to evaluate {model_folder}: {e}")
                # We can print a short traceback to help debug without spamming too much
                import traceback
                traceback.print_exc()
                
    else:
        # Single Run Mode
        run_evaluation_for_model(args, args.out_dir)


def run_evaluation_for_model(args, out_dir):
    # 1. Load Model
    # Note: load_pdlm_model uses get_base_dir() which reads os.environ["NANOCHAT_BASE_DIR"]
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
    
    os.makedirs(out_dir, exist_ok=True)
    
    all_raw_blocks = []
    generation_logs = []
    
    print(f"\nStarting Evaluation on {args.samples} samples...")
    # Use simple loop or manual pbar to avoid nesting issues if called multiple times? 
    # Tqdm is fine.
    pbar = tqdm(total=args.samples, desc="Processing")
    
    for i, (prompt_ids, ref_ids, raw_text) in enumerate(data_iter):
        # 3. Run Inference
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
            "prompt_ids": prompt_ids,
            "response": response_text
        })
        
        # 4. Parse Metrics
        blocks = parse_debug_into_blocks(block_debug)
        all_raw_blocks.extend(blocks)
        
        pbar.update(1)
        
    pbar.close()
    
    if not all_raw_blocks:
        print("No blocks were processed! Check dataset loading or model generation.")
        return
        
    # Save generations
    with open(os.path.join(out_dir, "generations.json"), "w") as f:
        json.dump(generation_logs, f, indent=2)

    print("Computing Statistics...")
    # 5. Compute Aggregate Stats
    stats = calculate_block_stats(all_raw_blocks)
    summary = aggregate_metrics(stats)
    
    print("\nEvaluation Summary:")
    print(json.dumps(summary, indent=2))
    
    # 6. Save & Visualize
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        
    plot_step_distribution(stats, out_dir)
    plot_multi_bucket_convergence(stats, out_dir)
    
    print("Generating trajectory plots for first 5 blocks...")
    for j in range(min(5, len(stats["block_durations"]))):
        plot_block_trajectory(stats, j, os.path.join(out_dir, "plots"))

    print(f"\nDone! Results saved to {out_dir}")

if __name__ == "__main__":
    main()
