# data/fineweb.py
#
# Loads FineWeb (sample-10BT) from HuggingFace, tokenizes with GPT-2 BPE,
# and caches as a binary shard for fast loading during training.

import json
import os
import time
from typing import Optional
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), ".cache")
DATASET_NAME = "HuggingFaceFW/fineweb"
DATASET_SPLIT = "sample-10BT"
SHARD_SIZE = 100_000_000  # tokens per shard
READY_FILENAME = "_fineweb_ready.json"
LOCK_FILENAME = ".fineweb_tokenize.lock"
LOCK_POLL_SECONDS = 5
LOCK_TIMEOUT_SECONDS = 4 * 60 * 60


def _ready_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, READY_FILENAME)


def _lock_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, LOCK_FILENAME)


def _parse_env_max_shards():
    raw = os.environ.get("FINEWEB_MAX_SHARDS")
    if raw is None or raw == "":
        return None
    return int(raw)


def _load_ready_metadata(cache_dir: str):
    path = _ready_path(cache_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _write_ready_metadata(cache_dir: str, shard_count: int, token_count: int):
    metadata = {
        "dataset_name": DATASET_NAME,
        "dataset_split": DATASET_SPLIT,
        "shard_count": shard_count,
        "token_count": token_count,
        "created_at": time.time(),
    }
    with open(_ready_path(cache_dir), "w") as f:
        json.dump(metadata, f)


def _list_shard_files(cache_dir: str, metadata=None):
    if metadata is not None:
        shard_count = metadata["shard_count"]
        return [
            os.path.join(cache_dir, f"shard_{idx:04d}.bin")
            for idx in range(shard_count)
        ]

    return sorted(
        [
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".bin")
        ]
    )


def tokenize_and_cache(cache_dir: str = DATA_CACHE_DIR, max_shards=None):
    """Download and tokenize FineWeb, saving binary shards to disk."""
    import tiktoken
    from datasets import load_dataset

    os.makedirs(cache_dir, exist_ok=True)
    max_shards = _parse_env_max_shards() if max_shards is None else max_shards
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # public API (50256 for GPT-2)

    shard_idx = 0
    total_tokens = 0
    token_buf = []

    print(f"Streaming {DATASET_NAME} ({DATASET_SPLIT})...")
    dataset = load_dataset(DATASET_NAME, DATASET_SPLIT, split="train", streaming=True)

    for doc in dataset:
        tokens = [eot] + enc.encode_ordinary(doc["text"])
        token_buf.extend(tokens)

        while len(token_buf) >= SHARD_SIZE:
            shard = np.array(token_buf[:SHARD_SIZE], dtype=np.uint16)
            path = os.path.join(cache_dir, f"shard_{shard_idx:04d}.bin")
            tmp_path = f"{path}.tmp"
            shard.tofile(tmp_path)
            os.replace(tmp_path, path)
            print(f"  Saved shard {shard_idx} ({SHARD_SIZE:,} tokens) -> {path}")
            token_buf = token_buf[SHARD_SIZE:]
            shard_idx += 1
            total_tokens += SHARD_SIZE
            if max_shards is not None and shard_idx >= max_shards:
                print(f"Reached max_shards={max_shards}, stopping.")
                _write_ready_metadata(cache_dir, shard_idx, total_tokens)
                return

    # Save remaining tokens as final shard
    if token_buf and (max_shards is None or shard_idx < max_shards):
        shard = np.array(token_buf, dtype=np.uint16)
        path = os.path.join(cache_dir, f"shard_{shard_idx:04d}.bin")
        tmp_path = f"{path}.tmp"
        shard.tofile(tmp_path)
        os.replace(tmp_path, path)
        print(f"  Saved final shard {shard_idx} ({len(token_buf):,} tokens) -> {path}")
        shard_idx += 1
        total_tokens += len(token_buf)

    _write_ready_metadata(cache_dir, shard_idx, total_tokens)


def ensure_tokenized_cache(
    cache_dir: str = DATA_CACHE_DIR,
    max_shards=None,
    timeout_seconds: int = LOCK_TIMEOUT_SECONDS,
):
    os.makedirs(cache_dir, exist_ok=True)
    lock_path = _lock_path(cache_dir)
    deadline = time.time() + timeout_seconds

    while True:
        metadata = _load_ready_metadata(cache_dir)
        if metadata is not None:
            return metadata

        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Timed out waiting for FineWeb cache lock: {lock_path}"
                )
            print("Waiting for FineWeb cache to finish building...")
            time.sleep(LOCK_POLL_SECONDS)
            continue

        try:
            with os.fdopen(fd, "w") as lock_file:
                lock_file.write(f"pid={os.getpid()} started_at={time.time()}\n")

            tokenize_and_cache(cache_dir=cache_dir, max_shards=max_shards)
            return _load_ready_metadata(cache_dir)
        finally:
            if os.path.exists(lock_path):
                os.remove(lock_path)


class FineWebDataset(Dataset):
    def __init__(self, cache_dir: str, seq_len: int = 1024):
        self.seq_len = seq_len
        metadata = _load_ready_metadata(cache_dir)
        shard_files = _list_shard_files(cache_dir, metadata=metadata)
        assert shard_files, (
            f"No .bin shards found in {cache_dir}. Run tokenize_and_cache() first."
        )

        # Load all shards into a single array
        arrays = [np.fromfile(f, dtype=np.uint16) for f in shard_files]
        self.data = np.concatenate(arrays)
        print(f"Loaded {len(self.data):,} tokens from {len(shard_files)} shard(s).")

        # Number of full (x, y) pairs
        self.n_samples = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


def get_dataloader(
    seq_len: int = 1024,
    micro_batch_size: int = 16,
    cache_dir: Optional[str] = None,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42,
) -> DataLoader:
    if cache_dir is None:
        cache_dir = DATA_CACHE_DIR
    if not os.path.isdir(cache_dir) or _load_ready_metadata(cache_dir) is None:
        print("Ensuring FineWeb cache is ready...")
        ensure_tokenized_cache(cache_dir)

    dataset = FineWebDataset(cache_dir, seq_len=seq_len)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator,
        drop_last=True,
    )


def download_data(cache_dir: str = DATA_CACHE_DIR, max_shards=None):
    if max_shards is None:
        max_shards = _parse_env_max_shards()
    metadata = ensure_tokenized_cache(cache_dir=cache_dir, max_shards=max_shards)
    if metadata is None:
        raise RuntimeError("FineWeb cache metadata missing after download.")
    return metadata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download and cache FineWeb shards.")
    parser.add_argument("--cache-dir", default=DATA_CACHE_DIR)
    parser.add_argument("--max-shards", type=int, default=None)
    args = parser.parse_args()

    metadata = download_data(cache_dir=args.cache_dir, max_shards=args.max_shards)
    if metadata is None:
        raise SystemExit("FineWeb cache metadata missing after download.")
    print(
        "FineWeb cache ready: "
        f"{metadata['shard_count']} shards, {metadata['token_count']:,} tokens"
    )
