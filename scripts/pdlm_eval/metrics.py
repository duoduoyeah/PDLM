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
    """
    block_durations = []
    token_convergence = [] # For each block, a list of convergence steps per token
    bucket_convergence = [] # For each block, the step where all tokens are stable
    first_stable_step = [] # For each block, the first step i where state[i] == state[i+1]
    token_probs_history = [] # For each block, a list of [step][pos] -> prob
    
    for b_idx, block in enumerate(blocks):
        # 1. Duration
        block_durations.append(len(block))
        
        # Get the final result of this block
        final_entry = block[-1]
        final_ids = final_entry["next_ids"][0].tolist() # (bucket_size,)
        bucket_size = len(final_ids)
        
        # Build history matrix: [step][pos]
        pred_history = []
        prob_history = []
        
        for step_entry in block:
            # pure_ids shape: (1, bucket, topk) -> we want top1: (bucket,)
            preds = step_entry["pure_ids"][0, :, 0].tolist() 
            probs = step_entry["pure_probs"][0, :, 0].tolist()
            
            pred_history.append(preds)
            prob_history.append(probs)
            
        token_probs_history.append(prob_history)
        
        # Calculate stability step for each position
        current_block_convergence = []
        for pos in range(bucket_size):
            target_token = final_ids[pos]
            stable_step = 0
            # Iterate backwards to find when it was LAST different
            for s in range(len(pred_history) - 1, -1, -1):
                if pred_history[s][pos] != target_token:
                    stable_step = s + 1
                    break
            current_block_convergence.append(stable_step)
            
        token_convergence.append(current_block_convergence)
        bucket_convergence.append(max(current_block_convergence))
        
        # Calculate First Stable Step (consecutive match)
        # Default to the last step if no consecutive match is found earlier
        stable_s = len(pred_history) - 1 
        for s in range(len(pred_history) - 1):
            if pred_history[s] == pred_history[s+1]:
                stable_s = s + 1 # Use s+1 to be consistent with 1-based step counting logic used elsewhere? 
                                 # Wait, usually steps are 0-indexed in history. 
                                 # convergence logic above returned s+1 which is effectively 1-based index or "count".
                                 # Let's stick to the step index. If at step 0 it matches step 1, we say stable at step 0?
                                 # The previous logic "stable_step = s + 1" means "after step s, it was stable".
                                 # So if pred[0] == target, stable_step = 0 (loop doesn't run).
                                 # Actually the loop runs: for s in range(len-1, -1, -1). 
                                 # If pred[last] == target, loop continues.
                                 # If pred[0] != target, stable_step = 0 + 1 = 1.
                                 # So convergence is 1-based count.
                                 
                # Let's match that: "After step X, it didn't change".
                # If pred[s] == pred[s+1], then at step `s+1` (1-based index `s+2`), we confirmed stability.
                # Example: pred[0] == pred[1] -> We confirmed at Step 2.
                stable_s = s + 2
                break
        first_stable_step.append(stable_s)
            
    return {
        "block_durations": block_durations,
        "token_convergence": token_convergence,
        "bucket_convergence": bucket_convergence,
        "first_stable_step": first_stable_step,
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
    avg_first_stable_step = np.mean(stats["first_stable_step"]) if stats["first_stable_step"] else 0
    
    return {
        "avg_steps_per_block": avg_steps,
        "avg_token_convergence": avg_token_convergence,
        "avg_bucket_convergence": avg_bucket_convergence,
        "avg_first_stable_step": avg_first_stable_step,
        "total_blocks": len(durations)
    }