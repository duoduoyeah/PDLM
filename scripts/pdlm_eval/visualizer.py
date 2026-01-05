
import os
import matplotlib.pyplot as plt
import numpy as np

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_block_trajectory(block_stats, block_idx, output_dir):
    """
    Plots the probability trajectory for each token in a specific block.
    """
    ensure_dir(output_dir)
    
    # block_stats is the dict returned by calculate_block_stats
    # We need:
    # - token_probs_history[block_idx] -> list of [step][pos] probs
    # - raw_blocks[block_idx] -> to get predictions and final tokens
    
    raw_block = block_stats["raw_blocks"][block_idx]
    probs_history = block_stats["token_probs_history"][block_idx] # [step][pos]
    
    # Get Final IDs (Ground Truth for this block)
    final_ids = raw_block[-1]["next_ids"][0].tolist()
    bucket_size = len(final_ids)
    
    # Extract prediction history to check for matches
    pred_history = []
    for entry in raw_block:
        pred_history.append(entry["pure_ids"][0, :, 0].tolist())
        
    steps = list(range(len(probs_history)))
    
    plt.figure(figsize=(10, 6))
    
    # Color map
    colors = plt.cm.jet(np.linspace(0, 1, bucket_size))
    
    for pos in range(bucket_size):
        # Extract series for this position
        p_series = [step_probs[pos] for step_probs in probs_history]
        match_series = [pred_history[s][pos] == final_ids[pos] for s in range(len(probs_history))]
        
        # Plot segments
        # We plot segment by segment to change style based on 'match'
        for i in range(len(steps) - 1):
            x = steps[i:i+2]
            y = p_series[i:i+2]
            is_match = match_series[i+1] # Status at end of segment
            
            linestyle = '-' if is_match else '--'
            alpha = 1.0 if is_match else 0.4
            
            plt.plot(x, y, color=colors[pos], linestyle=linestyle, alpha=alpha, 
                     label=f"Pos {pos}" if i == 0 else "")
            
    plt.title(f"Token Probability Trajectory (Block {block_idx})\nSolid=Final Token, Dashed=Temporary Token")
    plt.xlabel("Denoising Step")
    plt.ylabel("Probability of Argmax Token")
    plt.legend(loc='lower right', fontsize='small', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    save_path = os.path.join(output_dir, f"block_{block_idx}_traj.png")
    plt.savefig(save_path)
    plt.close()

def plot_multi_bucket_convergence(stats, output_dir, max_buckets=10):
    """
    Plots the convergence progress of multiple buckets.
    X-axis: Step index
    Y-axis: Number of tokens converged (matching final output)
    """
    ensure_dir(output_dir)
    
    plt.figure(figsize=(10, 6))
    
    # We'll use stats["token_convergence"] and stats["raw_blocks"]
    # Actually, it's easier to re-calculate matching tokens per step from raw_blocks
    
    num_to_plot = min(max_buckets, len(stats["raw_blocks"]))
    
    for b_idx in range(num_to_plot):
        raw_block = stats["raw_blocks"][b_idx]
        final_ids = raw_block[-1]["next_ids"][0].tolist()
        bucket_size = len(final_ids)
        
        steps = []
        converged_counts = []
        
        for s_idx, entry in enumerate(raw_block):
            preds = entry["pure_ids"][0, :, 0].tolist()
            matches = sum(1 for p, f in zip(preds, final_ids) if p == f)
            steps.append(s_idx)
            converged_counts.append(matches)
            
        plt.plot(steps, converged_counts, marker='o', markersize=4, alpha=0.7, label=f"Bucket {b_idx}")
        
    plt.title(f"Convergence Progress across Buckets (First {num_to_plot})")
    plt.xlabel("Denoising Step")
    plt.ylabel("Num Converged Tokens")
    plt.legend(loc='lower right', fontsize='x-small', ncol=2)
    plt.grid(True, alpha=0.3)
    plt.yticks(range(bucket_size + 1))
    
    save_path = os.path.join(output_dir, "multi_bucket_convergence.png")
    plt.savefig(save_path)
    plt.close()

def plot_step_distribution(stats, output_dir):
    ensure_dir(output_dir)
    durations = stats["block_durations"]
    
    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=range(min(durations), max(durations) + 2), align='left', rwidth=0.8)
    plt.title("Distribution of Steps per Block")
    plt.xlabel("Steps to Converge")
    plt.ylabel("Count")
    plt.grid(axis='y', alpha=0.3)
    
    save_path = os.path.join(output_dir, "steps_distribution.png")
    plt.savefig(save_path)
    plt.close()
