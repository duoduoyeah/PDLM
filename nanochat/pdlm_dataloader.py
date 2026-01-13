
from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer
from nanochat.sp_tokens.token_map import get_token_map, TokenMap

def tokenizing_distributed_data_loader_with_state(
    B,
    T,
    split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
    noise_total_steps=0,
    prefix_pure_tokens=0,
    model_type="pdlm",
    target_shift=1,
):
    """
    Stream pretraining text from parquet files, tokenize, yield training batches.

    This implementation became a bit more complex because we wish to support approximate resume training.
    Instead of turning this into a Class, we opt to return the state_dict with every batch,
    and then the caller can pass in a state_dict to resume training from a desired point.
    Note that this resumption is atm only *approximate* for simplicity.
    We won't repeat the same documents but we might skip a few.
    The state_dict that is returned can be later passed into this function via `resume_state_dict` to approximately resume.

    Perfect state resumption is possible but would be a lot more bloated, probably not worth it atm.

    model_type controls the input/target construction:
    - "pdlm" or "bd3lm": inputs are noisy versions of targets (for diffusion-based models)
    - "next_token_ar": inputs are shifted tokens (for autoregressive models like GPT)

    target_shift controls how many tokens ahead the target is for next_token_ar mode (1 = next-token).

    NOTE: this loader uses shard-based split logic (val is the last shard) and
    train includes all shards. A separate validation set can be used elsewhere
    and should be totally different from this shard-based eval.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    assert model_type in ["pdlm", "bd3lm", "next_token_ar"], f"model_type must be 'pdlm', 'bd3lm', or 'next_token_ar', got {model_type}"
    if model_type == "next_token_ar":
        assert target_shift >= 1, "target_shift must be >= 1 for next-token prediction"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files()
        if split == "val":
            parquet_paths = parquet_paths[-1:]
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            while pq_idx < len(parquet_paths): # iterate over all parquet files
                filepath = parquet_paths[pq_idx]
                pf = pq.ParquetFile(filepath)
                # Start from resume point if resuming on same file, otherwise from DDP rank
                # I know this state resumption is a little bit tricky and a little bit hacky... sigh.
                if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                    base_idx = resume_rg_idx // ddp_world_size # in units of ddp_world_size
                    base_idx += 1 # advance by 1 so that we definitely don't repeat data after resuming
                    rg_idx = base_idx * ddp_world_size + ddp_rank
                    if rg_idx >= pf.num_row_groups:
                        pq_idx += 1
                        continue
                    resume_rg_idx = None # set to None as we only want to do this a single time
                else:
                    rg_idx = ddp_rank
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column('text').to_pylist() # each batch is a parquet group, e.g. 1024 rows
                    # the tokenizer encode might want to go in even smaller batches, e.g. 128 rows
                    for i in range(0, len(batch), tokenizer_batch_size):
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx)
                    rg_idx += ddp_world_size # advance to the next row group (in DDP)
                pq_idx += 1 # advance to the next parquet file
            first_pass = False
    batches = document_batches()

    # Now emit batches of tokens.
    # For next_token_ar we need extra tokens for the shift, for pdlm/bd3lm we just need B*T
    if model_type == "next_token_ar":
        needed_tokens = B * T + target_shift
    elif model_type in ["pdlm", "bd3lm"]:
        needed_tokens = B * T
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # get the tokenizer and the bos token
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()

    # token_map is only needed for pdlm/bd3lm mode
    token_map: TokenMap = get_token_map(device="cpu") if model_type in ["pdlm", "bd3lm"] else None

    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx) = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
        # Move tokens from the deque into the scratch buffer
        tokens = [token_buffer.popleft() for _ in range(needed_tokens)]
        # CUDA supports memory pinning for asynchronous transfers between CPU and GPU
        use_cuda_optimizations = device == "cuda"

        if model_type == "next_token_ar":
            # Next-token autoregressive mode: inputs = tokens[:-shift], targets = tokens[shift:]
            scratch = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
            inputs_cpu = scratch[:-target_shift]
            targets_cpu = scratch[target_shift:]
            # Reshape to 2D and move to GPU async
            inputs = inputs_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)
            targets = targets_cpu.view(B, T).to(device=device, non_blocking=use_cuda_optimizations)

        elif model_type == "bd3lm":
            pass
            # BD3LM mode: TODO - implement bd3lm specific logic
            
            # we should first make sure how to make the noise
            
            # and then we will use this noise to noisy the input
            
            # but when compute the loss we need the noise design again 
            
            # targets is easy right
            
        elif model_type == "pdlm":
            # PDLM mode: inputs are noisy versions of targets
            # pick a random training step surrogate for noise scheduling if requested
            if noise_total_steps > 0:
                noise_step = torch.randint(1, noise_total_steps + 1, (B,))
            else:
                noise_step = 0

            targets_cpu = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
            targets_cpu = targets_cpu.view(B, T)

            noisy_levels = token_map.get_random_noisy_level(
                targets_cpu,
                step=noise_step,
                total_steps=noise_total_steps,
                prefix_pure_tokens=prefix_pure_tokens,
            )
            inputs_cpu = token_map.noise_tokens(targets_cpu, noisy_levels)

            # Reshape to 2D and move to GPU async
            inputs = inputs_cpu.to(device=device, non_blocking=use_cuda_optimizations)
            targets = targets_cpu.to(device=device, non_blocking=use_cuda_optimizations)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx} # we need this in case we wish to approximately resume training
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
