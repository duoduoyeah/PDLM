
from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer
from nanochat.sp_tokens.token_map import get_token_map, TokenMap
from nanochat.bd3lm_utils.bd3lm_mask import sample_t, q_xt, get_loss_scale, expand_block_to_seq

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
    bd3lm_block_size=1,
    bd3lm_mask_token_id=None,
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
    - "pdlm": inputs are noisy versions of targets using hierarchical token map
    - "bd3lm": inputs are masked versions of targets using MASK token
    - "next_token_ar": inputs are shifted tokens (for autoregressive models like GPT)

    target_shift controls:
    - For next_token_ar: how many tokens ahead the target is (1 = next-token)
    - For bd3lm: if >= 0, always mask position target_shift within each block (plus random masking)
                 if < 0, pure random masking based on sampled t

    block_size: block size for bd3lm mode (sequence is divided into blocks)
    mask_token_id: token id used for masking in bd3lm mode

    NOTE: train uses shard_*.parquet files, val uses validation_*.parquet files.

    Returns:
        inputs: (B, T) input token ids
        targets: (B, T) target token ids
        loss_extras: dict with model-specific loss info, or None
                     - For bd3lm: {"loss_scale": (B, T), "loss_mask": (B, T)}
                       NOTE: loss_mask indicates positions to compute loss, NOT input mask positions.
                       For target_shift mode, loss_mask only includes the target position (1 per block).
                     - For others: None
        state_dict: dict for resuming training
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    assert model_type in ["pdlm", "bd3lm", "next_token_ar"], f"model_type must be 'pdlm', 'bd3lm', or 'next_token_ar', got {model_type}"
    if model_type == "next_token_ar":
        assert target_shift >= 1, "target_shift must be >= 1 for next-token prediction"
    if model_type == "bd3lm":
        assert bd3lm_block_size >= 1, "block_size must be >= 1 for bd3lm"
        assert T % bd3lm_block_size == 0, f"T ({T}) must be divisible by block_size ({bd3lm_block_size})"
        assert bd3lm_mask_token_id is not None, "bd3lm_mask_token_id must be provided for bd3lm"
        if target_shift >= 0:
            assert 1 <= target_shift <= bd3lm_block_size, f"target_shift must be in [1, block_size] for bd3lm, got {target_shift}"

    # infinite iterator over document batches (list of text strings)
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    def document_batches():
        parquet_paths = list_parquet_files(split=split)
        resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
        resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
        resume_epoch = resume_state_dict.get("epoch", 0) if resume_state_dict is not None else 0
        first_pass = True
        pq_idx = resume_pq_idx # we kick off parquet files at the resume index (or by default just 0)
        epoch = resume_epoch
        while True: # iterate infinitely (multi-epoch)
            pq_idx = resume_pq_idx if first_pass else 0
            if not first_pass:
                epoch += 1  # increment epoch when restarting from first shard
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
                        yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
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

    # token_map is only needed for pdlm mode (bd3lm uses MASK token instead)
    token_map: TokenMap = get_token_map(device="cpu") if model_type == "pdlm" else None

    # num_blocks for bd3lm
    num_blocks = T // bd3lm_block_size if model_type == "bd3lm" else None

    # scratch buffer holds the tokens for one iteration
    token_buffer = deque() # we stream tokens on the right and pop from the left
    while True:
        # Accumulate enough tokens for one iteration before yielding.
        while len(token_buffer) < needed_tokens:
            doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
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
            loss_extras = None

        elif model_type == "bd3lm":
            # BD3LM mode: inputs are masked versions of targets using MASK token
            # targets are the clean tokens
            targets_cpu = torch.tensor(tokens, dtype=torch.long, pin_memory=use_cuda_optimizations)
            targets_cpu = targets_cpu.view(B, T)

            # Sample noise level t per block: shape (B, num_blocks)
            # t is sampled from [1/block_size, 1] to ensure at least 1 mask per block
            t = sample_t(
                batch_size=B,
                num_blocks=num_blocks,
                block_size=bd3lm_block_size,
                sampling_eps_max=1.0,
                device="cpu",
                antithetic_sampling=True,
            )

            # Determine forced mask position:
            # - target_shift >= 1: force position (target_shift-1) in each block (0-indexed)
            # - target_shift < 0: random position per block (normal BD3LM)
            forced_mask_position = (target_shift - 1) if target_shift >= 1 else None

            # Apply masking: q_xt converts clean tokens to noisy (masked) tokens
            # - First, one position per block is forced to be masked
            # - Then, remaining positions are masked with adjusted probability p'
            # inputs_cpu shape: (B, T), mask shape: (B, T)
            inputs_cpu, mask = q_xt(
                x0=targets_cpu,
                t=t,
                mask_token_id=bd3lm_mask_token_id,
                block_size=bd3lm_block_size,
                prefix_pure_tokens=prefix_pure_tokens,
                forced_mask_position=forced_mask_position,
            )

            # Compute loss_scale from t: shape (B, num_blocks) -> (B, T)
            loss_scale_per_block = get_loss_scale(t)  # (B, num_blocks)
            loss_scale = expand_block_to_seq(loss_scale_per_block, bd3lm_block_size)  # (B, T)

            # Create loss_mask: positions where loss is computed (not same as input mask for target_shift)
            if forced_mask_position is not None:
                # Target_shift mode: only compute loss at the forced position (1 per block)
                num_blocks = T // bd3lm_block_size
                loss_mask = torch.zeros_like(targets_cpu, dtype=torch.bool)
                for block_idx in range(num_blocks):
                    pos = block_idx * bd3lm_block_size + forced_mask_position
                    loss_mask[:, pos] = True
                # Exclude prefix_pure_tokens from loss
                if prefix_pure_tokens > 0:
                    loss_mask[:, :prefix_pure_tokens] = False
            else:
                # Normal mode: compute loss on all masked positions
                loss_mask = mask

            #TODO: for claude, what do you think we put the prefix_pure_tokens here?
            
            # Move to device
            inputs = inputs_cpu.to(device=device, non_blocking=use_cuda_optimizations)
            targets = targets_cpu.to(device=device, non_blocking=use_cuda_optimizations)
            loss_scale = loss_scale.to(device=device, non_blocking=use_cuda_optimizations)
            loss_mask = loss_mask.to(device=device, non_blocking=use_cuda_optimizations)
            loss_extras = {"loss_scale": loss_scale, "loss_mask": loss_mask}

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
            loss_extras = None

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch} # we need this in case we wish to approximately resume training
        yield inputs, targets, loss_extras, state_dict

def tokenizing_distributed_data_loader(*args, **kwargs):
    # helper function that only emits the inputs/targets and not the state_dict or loss_extras
    for inputs, targets, loss_extras, state_dict in tokenizing_distributed_data_loader_with_state(*args, **kwargs):
        yield inputs, targets
