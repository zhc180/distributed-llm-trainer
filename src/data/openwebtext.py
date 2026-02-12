"""OpenWebText dataloader utilities."""
import gzip
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, get_worker_info
from transformers import GPT2TokenizerFast


@dataclass
class OpenWebTextConfig:
    """Configuration for OpenWebText dataloader."""
    path: str
    seq_len: int
    tokenizer_name: str = "gpt2"
    max_tokens: Optional[int] = None
    streaming: bool = False
    cache_max_tokens: Optional[int] = None


class OpenWebTextDataset(Dataset):
    """Tokenized OpenWebText dataset sliced into fixed-length sequences."""

    def __init__(self, cfg: OpenWebTextConfig):
        self.cfg = cfg
        self.tokenizer = GPT2TokenizerFast.from_pretrained(cfg.tokenizer_name)

        if cfg.path.endswith(".gz"):
            with gzip.open(cfg.path, "rt", encoding="utf-8") as f:
                text = f.read()
        else:
            with open(cfg.path, "r", encoding="utf-8") as f:
                text = f.read()

        token_ids = self.tokenizer.encode(text)
        if cfg.max_tokens is not None:
            token_ids = token_ids[: cfg.max_tokens]

        if len(token_ids) < cfg.seq_len:
            raise ValueError(
                f"Not enough tokens ({len(token_ids)}) for seq_len={cfg.seq_len} in {cfg.path}"
            )

        self.tokens = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.cfg.seq_len

    def __getitem__(self, idx: int):
        start = idx * self.cfg.seq_len
        end = start + self.cfg.seq_len
        return (self.tokens[start:end],)


class OpenWebTextIterableDataset(IterableDataset):
    """Streaming OpenWebText dataset with optional tokenization cache."""

    def __init__(self, cfg: OpenWebTextConfig, rank: int = 0, world_size: int = 1):
        self.cfg = cfg
        self.rank = rank
        self.world_size = world_size
        self.tokenizer = GPT2TokenizerFast.from_pretrained(cfg.tokenizer_name)
        self._cache = OrderedDict()
        self._cache_tokens = 0

    def _open_file(self):
        if self.cfg.path.endswith(".gz"):
            return gzip.open(self.cfg.path, "rt", encoding="utf-8")
        return open(self.cfg.path, "r", encoding="utf-8")

    def _cache_tokens_for_line(self, line_idx: int, tokens: list) -> list:
        if not self.cfg.cache_max_tokens or self.cfg.cache_max_tokens <= 0:
            return tokens

        if line_idx in self._cache:
            self._cache.move_to_end(line_idx)
            return self._cache[line_idx]

        token_count = len(tokens)
        while self._cache and self._cache_tokens + token_count > self.cfg.cache_max_tokens:
            _, evicted = self._cache.popitem(last=False)
            self._cache_tokens -= len(evicted)

        if token_count <= self.cfg.cache_max_tokens:
            self._cache[line_idx] = tokens
            self._cache_tokens += token_count

        return tokens

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1

        shard_id = self.rank * num_workers + worker_id
        num_shards = self.world_size * num_workers

        buffer = []
        total_tokens = 0
        max_tokens = self.cfg.max_tokens

        with self._open_file() as f:
            for line_idx, line in enumerate(f):
                if line_idx % num_shards != shard_id:
                    continue

                tokens = self._cache_tokens_for_line(line_idx, self.tokenizer.encode(line))

                if max_tokens is not None:
                    remaining = max_tokens - total_tokens
                    if remaining <= 0:
                        break
                    if len(tokens) > remaining:
                        tokens = tokens[:remaining]

                total_tokens += len(tokens)
                buffer.extend(tokens)

                while len(buffer) >= self.cfg.seq_len:
                    chunk = buffer[: self.cfg.seq_len]
                    buffer = buffer[self.cfg.seq_len :]
                    yield torch.tensor(chunk, dtype=torch.long)

                if max_tokens is not None and total_tokens >= max_tokens:
                    break


def create_openwebtext_dataloader(
    path: str,
    batch_size: int,
    seq_len: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    tokenizer_name: str = "gpt2",
    max_tokens: Optional[int] = None,
    streaming: bool = False,
    cache_max_tokens: Optional[int] = None,
    num_workers: int = 2,
) -> DataLoader:
    """Create an OpenWebText dataloader with fixed-length token sequences."""
    cfg = OpenWebTextConfig(
        path=path,
        seq_len=seq_len,
        tokenizer_name=tokenizer_name,
        max_tokens=max_tokens,
        streaming=streaming,
        cache_max_tokens=cache_max_tokens,
    )
    if streaming:
        dataset = OpenWebTextIterableDataset(cfg, rank=rank, world_size=world_size)
        sampler = None
        shuffle = False
    else:
        dataset = OpenWebTextDataset(cfg)
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if distributed else None
        shuffle = sampler is None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )
