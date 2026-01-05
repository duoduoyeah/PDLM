
import torch
import numpy as np

def parse_debug_into_blocks(block_debug):
    """
    The model returns a flat list of debug entries. 
    This function groups them into 'blocks' (buckets) based on when a block finishes.
    
    Returns:
        list of lists: [ [entry_step0, entry_step1, ...], [next_block_step0, ...] ]
    """
    blocks = []
    current_block = []
    
    for entry in block_debug:
        current_block.append(entry)
        # "next_ids" is only present in the entry when the block is successfully denoised (pure)
        if "next_ids" in entry:
            blocks.append(current_block)
            current_block = []
            
    return blocks

def calculate_block_stats(blocks):
    """
    Computes detailed statistics for each block.
    
    Returns:
        dict: {
            "block_durations": list[int],
            "token_convergence": list[list[int]], # [block_idx][token_pos] -> step_converged
            "final_probs": list[list[float]],     # [block_idx][token_pos] -> final_confidence
            "raw_blocks": blocks                  # The original structured data
        }
    """
    block_durations = []
    token_convergence = [] # For each block, a list of convergence steps per token
    bucket_convergence = [] # For each block, the step where all tokens are stable
    token_probs_history = [] # For each block, a list of [step][pos] -> prob
    
    for b_idx, block in enumerate(blocks):
        # 1. Duration
        block_durations.append(len(block))
        
        # ... (rest of logic) ...
            
        token_convergence.append(current_block_convergence)
        bucket_convergence.append(max(current_block_convergence))
            
    return {
        "block_durations": block_durations,
        "token_convergence": token_convergence,
        "bucket_convergence": bucket_convergence,
        "token_probs_history": token_probs_history,
        "raw_blocks": blocks
    }

def aggregate_metrics(stats):
    """
    Returns summary scalar metrics.
    """
    durations = stats["block_durations"]
    if not durations:
        return {}
        
    avg_steps = np.mean(durations)
    
    # Flatten convergence info
    all_convergence = [s for sublist in stats["token_convergence"] for s in sublist]
    avg_token_convergence = np.mean(all_convergence) if all_convergence else 0
    
    avg_bucket_convergence = np.mean(stats["bucket_convergence"]) if stats["bucket_convergence"] else 0
    
    return {
        "avg_steps_per_block": avg_steps,
        "avg_token_convergence": avg_token_convergence,
        "avg_bucket_convergence": avg_bucket_convergence,
        "total_blocks": len(durations)
    }
