"""
Common utilities for dataloaders.
"""

from collections import deque

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer


def create_document_batches(split, resume_state_dict, tokenizer_batch_size=128):
    """
    Infinite iterator over document batches from parquet files.

    Yields:
        doc_batch: list of text strings
        pq_idx: parquet file index
        rg_idx: row group index
        epoch: current epoch number
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    parquet_paths = list_parquet_files(split=split)
    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 0) if resume_state_dict is not None else 0
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
        pq_idx = resume_pq_idx if first_pass else 0
        if not first_pass:
            epoch += 1
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 to not repeat data
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], pq_idx, rg_idx, epoch
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False


class TokenBuffer:
    """
    Buffer that accumulates tokens from document batches.
    """
    def __init__(self, batches_iter, tokenizer, bos_token, tokenizer_threads=4):
        self.batches_iter = batches_iter
        self.tokenizer = tokenizer
        self.bos_token = bos_token
        self.tokenizer_threads = tokenizer_threads
        self.buffer = deque()
        self.last_pq_idx = 0
        self.last_rg_idx = 0
        self.last_epoch = 0

    def get_tokens(self, needed_tokens):
        """
        Get exactly `needed_tokens` tokens from the buffer.
        Returns tokens list and current state (pq_idx, rg_idx, epoch).
        """
        while len(self.buffer) < needed_tokens:
            doc_batch, pq_idx, rg_idx, epoch = next(self.batches_iter)
            self.last_pq_idx = pq_idx
            self.last_rg_idx = rg_idx
            self.last_epoch = epoch
            token_lists = self.tokenizer.encode(
                doc_batch, prepend=self.bos_token, num_threads=self.tokenizer_threads
            )
            for tokens in token_lists:
                self.buffer.extend(tokens)

        tokens = [self.buffer.popleft() for _ in range(needed_tokens)]
        return tokens, self.last_pq_idx, self.last_rg_idx, self.last_epoch


def create_token_buffer(split, resume_state_dict, tokenizer_threads=4, tokenizer_batch_size=128):
    """
    Create a TokenBuffer for the given split.
    """
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    batches_iter = create_document_batches(split, resume_state_dict, tokenizer_batch_size)
    return TokenBuffer(batches_iter, tokenizer, bos_token, tokenizer_threads)
