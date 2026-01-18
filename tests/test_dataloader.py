"""
Test dataset and dataloader. Example run:

python -m pytest tests/test_dataloader.py -v
"""

import os
import pytest
from nanochat.dataset import list_parquet_files, DATA_DIR, MAX_SHARD, MAX_VAL_SHARD


class TestListParquetFiles:
    """Test list_parquet_files function with split parameter."""

    def test_train_split_returns_shard_files(self):
        """Train split should only return shard_*.parquet files."""
        train_files = list_parquet_files(split="train")
        for f in train_files:
            basename = os.path.basename(f)
            assert basename.startswith("shard_"), f"Expected shard_* file, got {basename}"
            assert basename.endswith(".parquet"), f"Expected .parquet file, got {basename}"

    def test_val_split_returns_validation_files(self):
        """Val split should only return validation_*.parquet files."""
        val_files = list_parquet_files(split="val")
        for f in val_files:
            basename = os.path.basename(f)
            assert basename.startswith("validation_"), f"Expected validation_* file, got {basename}"
            assert basename.endswith(".parquet"), f"Expected .parquet file, got {basename}"

    def test_train_and_val_are_disjoint(self):
        """Train and val files should have no overlap."""
        train_files = set(list_parquet_files(split="train"))
        val_files = set(list_parquet_files(split="val"))
        overlap = train_files & val_files
        assert len(overlap) == 0, f"Train and val overlap: {overlap}"

    def test_invalid_split_raises(self):
        """Invalid split should raise AssertionError."""
        with pytest.raises(AssertionError):
            list_parquet_files(split="invalid")

    def test_default_split_is_train(self):
        """Default split should be train."""
        default_files = list_parquet_files()
        train_files = list_parquet_files(split="train")
        assert default_files == train_files


def _tokenizer_available():
    """Check if tokenizer is available."""
    try:
        from nanochat.tokenizer import get_tokenizer
        get_tokenizer()
        return True
    except (FileNotFoundError, ImportError):
        return False


class TestDataloaderSplits:
    """Test that dataloader uses correct files for each split."""

    @pytest.mark.skipif(not _tokenizer_available(), reason="Tokenizer not available")
    def test_train_loader_yields_batches(self):
        """Train dataloader should yield batches from shard_* files."""
        train_files = list_parquet_files(split="train")
        if len(train_files) == 0:
            pytest.skip("No train parquet files available")

        from nanochat.dataloader import tokenizing_distributed_data_loader_with_state

        loader = tokenizing_distributed_data_loader_with_state(
            B=2,
            T=64,
            split="train",
            device="cpu",
            model_type="next_token_ar",
            target_shift=1,
        )
        # Just verify we can get one batch
        inputs, targets, loss_extras, state_dict = next(loader)
        assert inputs.shape == (2, 64), f"Expected shape (2, 64), got {inputs.shape}"
        assert targets.shape == (2, 64), f"Expected shape (2, 64), got {targets.shape}"

    @pytest.mark.skipif(not _tokenizer_available(), reason="Tokenizer not available")
    def test_val_loader_yields_batches(self):
        """Val dataloader should yield batches from validation_* files."""
        val_files = list_parquet_files(split="val")
        if len(val_files) == 0:
            pytest.skip("No validation parquet files available")

        from nanochat.dataloader import tokenizing_distributed_data_loader_with_state

        loader = tokenizing_distributed_data_loader_with_state(
            B=2,
            T=64,
            split="val",
            device="cpu",
            model_type="next_token_ar",
            target_shift=1,
        )
        # Just verify we can get one batch
        inputs, targets, loss_extras, state_dict = next(loader)
        assert inputs.shape == (2, 64), f"Expected shape (2, 64), got {inputs.shape}"
        assert targets.shape == (2, 64), f"Expected shape (2, 64), got {targets.shape}"


class TestDataConstants:
    """Test that data constants are correct."""

    def test_max_shard_is_9(self):
        """MAX_SHARD should be 9 (indices 0-9)."""
        assert MAX_SHARD == 9, f"Expected MAX_SHARD=9, got {MAX_SHARD}"

    def test_max_val_shard_is_9(self):
        """MAX_VAL_SHARD should be 9 (indices 0-9)."""
        assert MAX_VAL_SHARD == 9, f"Expected MAX_VAL_SHARD=9, got {MAX_VAL_SHARD}"
