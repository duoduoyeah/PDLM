"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The URL on the internet where the data is hosted and downloaded from on demand
DATASET_REPO = "duoduoyeah/simple-story-shuffle"
REPO_PATH = "data"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/main/{REPO_PATH}"
# Training shards: shard_00000.parquet to shard_00009.parquet
MAX_SHARD = 9  # last index (0-indexed)
train_index_to_filename = lambda index: f"shard_{index:05d}.parquet"
# Validation shards: validation_00000.parquet to validation_00009.parquet
MAX_VAL_SHARD = 9  # last index (0-indexed)
val_index_to_filename = lambda index: f"validation_{index:05d}.parquet"
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "simple_story_data")
os.makedirs(DATA_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported

def list_parquet_files(split="train", data_dir=None):
    """
    Looks into a data dir and returns full paths to parquet files for the given split.

    Args:
        split: "train" for shard_*.parquet, "val" for validation_*.parquet
        data_dir: directory to look in (default: DATA_DIR)
    """
    assert split in ["train", "val"], f"split must be 'train' or 'val', got {split}"
    data_dir = DATA_DIR if data_dir is None else data_dir
    prefix = "shard_" if split == "train" else "validation_"
    parquet_files = sorted([
        f for f in os.listdir(data_dir)
        if f.startswith(prefix) and f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths

def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val".
      train uses shard_*.parquet files, val uses validation_*.parquet files.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(split=split)
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts

# -----------------------------------------------------------------------------
def download_single_file(args):
    """
    Downloads a single file, with some backoff.

    Args:
        args: tuple of (index, split) where split is "train" or "val"
    """
    index, split = args
    index_to_filename = train_index_to_filename if split == "train" else val_index_to_filename

    # Construct the local filepath for this file and skip if it already exists
    filename = index_to_filename(index)
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download dataset shards (train and/or validation)")
    parser.add_argument("-n", "--num-files", type=int, default=-1, help="Number of shards per split to download (-1 = all)")
    parser.add_argument("-w", "--num-workers", type=int, default=4, help="Number of parallel download workers (default: 4)")
    parser.add_argument("--split", type=str, default="both", choices=["train", "val", "both"],
                        help="Which split to download: train, val, or both (default: both)")
    args = parser.parse_args()

    print("Dataset download info:")
    print(f"  Dataset repo: {DATASET_REPO}")
    print(f"  Base URL: {BASE_URL}")
    print(f"  Target directory: {DATA_DIR}")
    print()

    # Build list of (index, split) tuples to download
    download_args = []
    if args.split in ["train", "both"]:
        num_train = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
        download_args.extend([(i, "train") for i in range(num_train)])
        print(f"  Train shards: {train_index_to_filename(0)} .. {train_index_to_filename(MAX_SHARD)} ({num_train} files)")
    if args.split in ["val", "both"]:
        num_val = MAX_VAL_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_VAL_SHARD + 1)
        download_args.extend([(i, "val") for i in range(num_val)])
        print(f"  Val shards: {val_index_to_filename(0)} .. {val_index_to_filename(MAX_VAL_SHARD)} ({num_val} files)")

    print()
    print(f"Downloading {len(download_args)} files using {args.num_workers} workers...")
    print()

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, download_args)

    # Report results
    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(download_args)} files to {DATA_DIR}")
