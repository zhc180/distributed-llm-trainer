"""OpenWebText dataloader utilities."""
import gzip
from dataclasses import dataclass
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import GPT2TokenizerFast


@dataclass
class OpenWebTextConfig:
    """Configuration for OpenWebText dataloader."""
    path: str
    seq_len: int
    tokenizer_name: str = "gpt2"
    max_tokens: Optional[int] = None


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


def create_openwebtext_dataloader(
    path: str,
    batch_size: int,
    seq_len: int,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    tokenizer_name: str = "gpt2",
    max_tokens: Optional[int] = None,
    num_workers: int = 2,
) -> DataLoader:
    """Create an OpenWebText dataloader with fixed-length token sequences."""
    cfg = OpenWebTextConfig(
        path=path,
        seq_len=seq_len,
        tokenizer_name=tokenizer_name,
        max_tokens=max_tokens,
    )
    dataset = OpenWebTextDataset(cfg)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if distributed else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
    )
