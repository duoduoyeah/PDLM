#!/usr/bin/env python3
"""
Dump the validation dataset rows as string representations.
Uses the same filtering logic as the eval pipeline.
"""
import argparse
import os
from datasets import load_dataset
from nanochat.tokenizer import get_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Dump validation dataset rows.")
    parser.add_argument("--dataset", type=str, default="duoduoyeah/simple-story-shuffle", help="HF Dataset name")
    parser.add_argument("--dataset-config", type=str, default=None, help="HF Dataset config")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split")
    parser.add_argument("--prefix-len", type=int, default=256, help="Prefix length for filtering")
    parser.add_argument("--output", type=str, default="val_dataset_dump.txt", help="Output file path")
    parser.add_argument("--max-samples", type=int, default=32, help="Max samples to dump")
    
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset}/{args.dataset_config} split={args.split}")
    try:
        ds = load_dataset(args.dataset, args.dataset_config, split=args.split, streaming=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize tokenizer to filter rows by length exactly as the eval pipeline does
    tokenizer = get_tokenizer()
    bos_id = tokenizer.get_bos_token_id()

    print(f"Dumping to {args.output}...")
    
    count = 0
    with open(args.output, "w", encoding="utf-8") as f:
        for row in ds:
            text = row.get("text", "")
            if not text.strip():
                continue

            ids = tokenizer.encode(text, prepend=bos_id)
            
            # Filter logic from scripts/pdlm_eval/data_loader.py
            if len(ids) <= args.prefix_len + 10:
                continue

            # Write string representation
            f.write(repr(text) + "\n")
            
            count += 1
            if args.max_samples is not None and count >= args.max_samples:
                break
                
    print(f"Dumped {count} samples.")

if __name__ == "__main__":
    main()
