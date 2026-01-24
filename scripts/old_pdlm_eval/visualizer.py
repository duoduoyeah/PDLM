
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

def plot_average_convergence(stats, output_dir):
    """
    Plots the average convergence progress across ALL blocks.
    X-axis: Step index
    Y-axis: Number of tokens converged
    Includes mean line and percentile ranges (25-75% and 10-90%).
    """
    ensure_dir(output_dir)
    
    raw_blocks = stats["raw_blocks"]
    if not raw_blocks:
        return

    # 1. Determine max steps and bucket size
    max_steps = max(len(b) for b in raw_blocks)
    
    # We assume bucket size is generally constant, take from first block
    first_final = raw_blocks[0][-1]["next_ids"][0]
    bucket_size = first_final.numel()
    
    # 2. Collect convergence data per step
    # convergence_data[step] = list of [num_converged for each block at this step]
    convergence_data = [[] for _ in range(max_steps)]
    
    for block in raw_blocks:
        final_ids = block[-1]["next_ids"][0].tolist()
        
        for step_idx, entry in enumerate(block):
            preds = entry["pure_ids"][0, :, 0].tolist()
            # Count matches
            matches = sum(1 for p, f in zip(preds, final_ids) if p == f)
            convergence_data[step_idx].append(matches)
            
        # If block finished early, assume it stays perfectly converged for remaining steps?
        # Typically "average convergence" implies "at step X". 
        # If a block finishes at step 3, at step 4 it is technically "done" (max matches).
        # To make the graph smoother and represent "state of the world", we should fill forward.
        final_matches = bucket_size # By definition, if block finished, it matches
        for step_idx in range(len(block), max_steps):
            convergence_data[step_idx].append(final_matches)

    # 3. Calculate Stats
    steps = np.arange(max_steps)
    means = []
    p10s, p25s, p75s, p90s = [], [], [], []
    
    for step_vals in convergence_data:
        if not step_vals:
            means.append(0)
            p10s.append(0); p25s.append(0); p75s.append(0); p90s.append(0)
            continue
            
        means.append(np.mean(step_vals))
        p10s.append(np.percentile(step_vals, 10))
        p25s.append(np.percentile(step_vals, 25))
        p75s.append(np.percentile(step_vals, 75))
        p90s.append(np.percentile(step_vals, 90))
        
    # 4. Plot
    plt.figure(figsize=(10, 6))
    
    # Ranges
    plt.fill_between(steps, p10s, p90s, color='blue', alpha=0.1, label='10th-90th Percentile')
    plt.fill_between(steps, p25s, p75s, color='blue', alpha=0.2, label='25th-75th Percentile')
    
    # Mean line
    plt.plot(steps, means, color='blue', linewidth=2, label='Average Convergence')
    
    plt.title(f"Average Token Convergence vs Step (Across {len(raw_blocks)} Blocks)")
    plt.xlabel("Denoising Step")
    plt.ylabel(f"Converged Tokens (Max {bucket_size})")
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.yticks(range(bucket_size + 1))
    
    save_path = os.path.join(output_dir, "average_convergence.png")
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
