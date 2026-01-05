
import torch
from datasets import load_dataset
from nanochat.tokenizer import get_tokenizer

def get_eval_data_iterator(dataset_name="duoduoyeah/TinyStories", dataset_config=None, split="validation", max_samples=None, prefix_len=16):
    """
    Yields tuples of (prompt_ids, reference_ids)
    
    Args:
        dataset_name: HF dataset name
        dataset_config: HF dataset config/subset
        split: 'train', 'validation', or 'test'
        max_samples: Stop after this many samples
        prefix_len: Number of tokens to use as the prompt
    """
    print(f"Loading dataset: {dataset_name}/{dataset_config} split={split}")
    try:
        ds = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    tokenizer = get_tokenizer()
    bos_id = tokenizer.get_bos_token_id()
    
    count = 0
    for row in ds:
        text = row.get("text", "")
        if not text.strip():
            continue
            
        # Tokenize
        # Note: We don't prepend BOS here strictly because we might be grabbing mid-sentence chunks
        # but for consistency with training, let's prepend BOS if it's the start of a doc.
        ids = tokenizer.encode(text, prepend=bos_id)
        
        # We need at least prefix_len + some tokens for generation
        if len(ids) <= prefix_len + 10:
            continue
            
        prompt_ids = ids[:prefix_len]
        reference_ids = ids[prefix_len:] # The rest of the document/sequence
        
        yield prompt_ids, reference_ids, text
        
        count += 1
        if max_samples is not None and count >= max_samples:
            break
