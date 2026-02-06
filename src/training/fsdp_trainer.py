"""Fully Sharded Data Parallel (FSDP) Trainer.

FSDP shards model parameters, gradients, and optimizer states across GPUs.
This allows training much larger models than DDP.

Key concepts:
- All-Gather: Collect full parameters before forward/backward
- Reduce-Scatter: Distribute gradient shards after backward
- CPU Offloading: Offload optimizer states to CPU memory

Usage:
    torchrun --nproc_per_node=4 fsdp_trainer.py --config configs/medium_model.yaml

Requirements:
    - PyTorch >= 2.0
    - Multiple GPUs recommended
"""
import argparse
import functools
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, Set

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullStateDictConfig,
    StateDictType,
)
from torch.utils.data import DataLoader, DistributedSampler

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt import GPT, TransformerBlock, count_parameters
from models.config import GPTConfig


@dataclass
class FSDPConfig:
    """FSDP-specific configuration."""
    # Sharding strategy
    # FULL_SHARD: Shard everything (ZeRO-3)
    # SHARD_GRAD_OP: Shard gradients and optimizer (ZeRO-2)
    # NO_SHARD: No sharding (like DDP)
    sharding_strategy: str = "FULL_SHARD"

    # CPU offloading
    cpu_offload: bool = False

    # Mixed precision policy
    mixed_precision: str = "bf16"

    # Backward prefetch
    backward_prefetch: str = "BACKWARD_PRE"

    # Activation checkpointing
    activation_checkpointing: bool = True

    # Limit all-gathers (memory optimization)
    limit_all_gathers: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 4
    max_seq_len: int = 1024
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    max_steps: int = 10000
    warmup_steps: int = 1000
    log_interval: int = 10
    save_interval: int = 1000
    gradient_accumulation_steps: int = 8
    checkpoint_dir: str = "checkpoints_fsdp"


class FSDPTrainer:
    """FSDP Trainer for large-scale distributed training."""

    def __init__(
        self,
        model_config: GPTConfig,
        training_config: TrainingConfig,
        fsdp_config: FSDPConfig,
    ):
        self.model_config = model_config
        self.training_config = training_config
        self.fsdp_config = fsdp_config

        # Initialize distributed
        self._setup_distributed()

        # Setup device
        self._setup_device()

        # Build and wrap model with FSDP
        self._setup_model()

        # Setup optimizer
        self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.tokens_seen = 0

    def _setup_distributed(self):
        """Initialize distributed environment."""
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.is_main_process = self.rank == 0

        if self.is_main_process:
            print(f"FSDP Training initialized")
            print(f"World size: {self.world_size}")
            print(f"Sharding strategy: {self.fsdp_config.sharding_strategy}")

    def _setup_device(self):
        """Setup CUDA device."""
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.device)

        # Setup dtype for autocast
        mp = self.fsdp_config.mixed_precision
        if mp == "bf16":
            self.compute_dtype = torch.bfloat16
        elif mp == "fp16":
            self.compute_dtype = torch.float16
        else:
            self.compute_dtype = torch.float32

        if self.is_main_process:
            print(f"Device: {self.device}")
            print(f"Compute dtype: {self.compute_dtype}")

    def _get_fsdp_policy(self):
        """Create FSDP wrapping policy.

        TODO: Implement FSDP wrapping policy

        FSDP needs to know HOW to shard your model. The wrapping policy
        determines which modules become separate FSDP units.

        For transformers, we wrap each TransformerBlock separately because:
        1. Each block is a logical unit with similar size
        2. Allows all-gather/reduce-scatter at block boundaries
        3. Standard practice in production systems

        Use transformer_auto_wrap_policy with transformer_layer_cls={TransformerBlock}

        Implementation:
            return functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={<your transformer block class>},
            )

        Alternative: size_based_auto_wrap_policy wraps modules above a size threshold.
        This is more general but less optimal for transformers.

        Question to think about: What happens if you wrap too granularly (e.g., every Linear)?
        Answer: Too much communication overhead from frequent all-gather/reduce-scatter.
        """
        raise NotImplementedError("Implement FSDP wrapping policy")

    def _get_mixed_precision_policy(self) -> Optional[MixedPrecision]:
        """Create mixed precision policy for FSDP.

        TODO: Implement mixed precision policy

        Mixed precision trains with lower precision (FP16/BF16) to:
        1. Reduce memory usage (~2x)
        2. Speed up computation (tensor cores)
        3. Reduce communication bandwidth

        FSDP MixedPrecision has three dtype settings:
        - param_dtype: Dtype for parameters during forward/backward
        - reduce_dtype: Dtype for gradient reduction (all-reduce/reduce-scatter)
        - buffer_dtype: Dtype for buffers (e.g., running stats in BatchNorm)

        BF16 vs FP16:
        - BF16: Same range as FP32, less precision. No loss scaling needed!
        - FP16: More precision, smaller range. Needs GradScaler for stability.

        For BF16, use torch.bfloat16 for all three dtypes.
        For FP16, use torch.float16 for all three dtypes.
        For FP32 (or unknown), return None (no mixed precision).

        Implementation:
            if mp == "bf16":
                return MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                )
            # ... similar for fp16
            # Return None for fp32

        Question: Why might you use different dtypes for param vs reduce?
        Answer: Sometimes FP32 reduction improves gradient precision while
                keeping fast FP16 compute for forward/backward.
        """
        mp = self.fsdp_config.mixed_precision

        raise NotImplementedError("Implement mixed precision policy")

    def _get_sharding_strategy(self) -> ShardingStrategy:
        """Get sharding strategy from config.

        TODO: Implement sharding strategy mapping

        FSDP supports different sharding strategies (ZeRO stages):

        FULL_SHARD (ZeRO-3):
            - Shards: parameters + gradients + optimizer states
            - Memory: O(model_size / num_gpus) per GPU
            - Communication: All-gather before forward AND backward, reduce-scatter after backward
            - Best for: Very large models that don't fit on single GPU

        SHARD_GRAD_OP (ZeRO-2):
            - Shards: gradients + optimizer states (NOT parameters)
            - Memory: Parameters replicated, but optimizer memory is sharded
            - Communication: Less than FULL_SHARD (no param all-gather)
            - Best for: Models that fit in memory but optimizer states don't

        NO_SHARD:
            - No sharding, like regular DDP
            - Memory: Full model + optimizer on each GPU
            - Best for: Small models, baseline comparison

        HYBRID_SHARD:
            - FULL_SHARD within a node, replicate across nodes
            - Reduces inter-node communication
            - Best for: Multi-node training

        Implementation:
            Create a dict mapping string names to ShardingStrategy enum values,
            return the one matching self.fsdp_config.sharding_strategy

        Hint: ShardingStrategy.FULL_SHARD, ShardingStrategy.SHARD_GRAD_OP, etc.
        """
        raise NotImplementedError("Implement sharding strategy mapping")

    def _setup_model(self):
        """Build and wrap model with FSDP."""
        # Create model on meta device first (for large models)
        # This avoids OOM when initializing on GPU
        if self.is_main_process:
            print("Building model...")

        # For now, create on CPU then move
        model = GPT(self.model_config)

        if self.is_main_process:
            print(f"Model parameters: {count_parameters(model):,}")

        # FSDP configuration
        fsdp_kwargs = {
            "auto_wrap_policy": self._get_fsdp_policy(),
            "sharding_strategy": self._get_sharding_strategy(),
            "mixed_precision": self._get_mixed_precision_policy(),
            "device_id": self.local_rank,
            "limit_all_gathers": self.fsdp_config.limit_all_gathers,
        }

        # CPU offloading
        if self.fsdp_config.cpu_offload:
            fsdp_kwargs["cpu_offload"] = CPUOffload(offload_params=True)

        # Backward prefetch
        if self.fsdp_config.backward_prefetch == "BACKWARD_PRE":
            fsdp_kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_PRE
        elif self.fsdp_config.backward_prefetch == "BACKWARD_POST":
            fsdp_kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_POST

        # Wrap with FSDP
        self.model = FSDP(model, **fsdp_kwargs)

        # Apply activation checkpointing if enabled
        if self.fsdp_config.activation_checkpointing:
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                checkpoint_wrapper,
                CheckpointImpl,
                apply_activation_checkpointing,
            )

            check_fn = lambda submodule: isinstance(submodule, TransformerBlock)
            apply_activation_checkpointing(
                self.model,
                checkpoint_wrapper_fn=checkpoint_wrapper,
                check_fn=check_fn,
            )

            if self.is_main_process:
                print("Activation checkpointing enabled")

        if self.is_main_process:
            # Print FSDP sharding info
            print(f"FSDP sharding strategy: {self._get_sharding_strategy()}")

    def _setup_optimizer(self):
        """Setup optimizer."""
        cfg = self.training_config

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            weight_decay=cfg.weight_decay,
        )

    def get_lr(self, step: int) -> float:
        """Cosine learning rate with warmup."""
        import math
        cfg = self.training_config

        if step < cfg.warmup_steps:
            return cfg.learning_rate * step / cfg.warmup_steps

        decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
        decay_ratio = min(decay_ratio, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return cfg.learning_rate * 0.1 + coeff * (cfg.learning_rate * 0.9)

    def train_step(self, batch: dict) -> dict:
        """Training step with gradient accumulation."""
        cfg = self.training_config

        self.model.train()
        total_loss = 0.0

        input_ids = batch["input_ids"].to(self.device)
        labels = input_ids.clone()

        # Gradient accumulation
        for micro_step in range(cfg.gradient_accumulation_steps):
            micro_bs = input_ids.shape[0] // cfg.gradient_accumulation_steps
            start = micro_step * micro_bs
            end = start + micro_bs

            micro_input = input_ids[start:end]
            micro_labels = labels[start:end]

            # Forward with autocast
            with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
                _, loss = self.model(micro_input, labels=micro_labels)
                loss = loss / cfg.gradient_accumulation_steps

            # Backward
            loss.backward()
            total_loss += loss.item()

        # Gradient clipping (FSDP-compatible)
        if cfg.grad_clip > 0:
            self.model.clip_grad_norm_(cfg.grad_clip)

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Update LR
        lr = self.get_lr(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        # Update counters
        self.global_step += 1
        self.tokens_seen += input_ids.numel() * self.world_size

        return {"loss": total_loss, "lr": lr, "tokens": self.tokens_seen}

    def save_checkpoint(self, path: str):
        """Save FSDP checkpoint.

        TODO: Implement FSDP checkpoint saving

        FSDP checkpointing is tricky because the model is SHARDED across GPUs!
        Each GPU only has a piece of the model.

        Options for saving:
        1. FULL_STATE_DICT: Gather all shards to rank 0, save single file
           - Pros: Standard format, easy to load for inference
           - Cons: Requires memory for full model on rank 0

        2. SHARDED_STATE_DICT: Each rank saves its shard
           - Pros: No memory spike, faster
           - Cons: Must load with same world_size

        For this exercise, use FULL_STATE_DICT (simpler):

        Steps:
        1. Create directory (only on main process)

        2. Configure state dict type:
           save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
           - offload_to_cpu: Move gathered state to CPU (saves GPU memory)
           - rank0_only: Only rank 0 gets the full state dict

        3. Use context manager to gather state:
           with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
               model_state = self.model.state_dict()
               optim_state = FSDP.optim_state_dict(self.model, self.optimizer)

        4. Save on main process only (rank 0):
           - Include: model_state, optim_state, global_step, tokens_seen, configs

        5. Synchronize all processes with dist.barrier()
           - Why? Ensures all ranks finish before continuing
           - Prevents race conditions

        Hint: Check self.is_main_process before saving/printing
        """
        raise NotImplementedError("Implement FSDP checkpoint saving")

    def load_checkpoint(self, path: str):
        """Load FSDP checkpoint."""
        # Load on rank 0 and broadcast
        if self.is_main_process:
            checkpoint = torch.load(path, map_location="cpu")
        else:
            checkpoint = None

        # Broadcast checkpoint
        checkpoint = [checkpoint]
        dist.broadcast_object_list(checkpoint, src=0)
        checkpoint = checkpoint[0]

        # Load model state
        with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
            self.model.load_state_dict(checkpoint["model"])

        # Load optimizer state
        optim_state = FSDP.optim_state_dict_to_load(
            self.model, self.optimizer, checkpoint["optimizer"]
        )
        self.optimizer.load_state_dict(optim_state)

        self.global_step = checkpoint["global_step"]
        self.tokens_seen = checkpoint["tokens_seen"]

        if self.is_main_process:
            print(f"Loaded checkpoint from {path} (step {self.global_step})")

    def get_memory_stats(self) -> dict:
        """Get GPU memory statistics."""
        if not torch.cuda.is_available():
            return {}

        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }


def create_dummy_dataloader(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_batches: int = 1000,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create dummy dataloader for testing."""
    data = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len))
    dataset = torch.utils.data.TensorDataset(data)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True,
        num_workers=4,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", default="medium", choices=["small", "medium", "large", "xl"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--sharding", default="FULL_SHARD", choices=["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD"])
    parser.add_argument("--cpu_offload", action="store_true")
    parser.add_argument("--no_activation_checkpointing", action="store_true")
    args = parser.parse_args()

    # Model config
    configs = {
        "small": GPTConfig.gpt2_small,
        "medium": GPTConfig.gpt2_medium,
        "large": GPTConfig.gpt2_large,
        "xl": GPTConfig.gpt2_xl,
    }
    model_config = configs[args.model_size]()

    # Training config
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
    )

    # FSDP config
    fsdp_config = FSDPConfig(
        sharding_strategy=args.sharding,
        cpu_offload=args.cpu_offload,
        activation_checkpointing=not args.no_activation_checkpointing,
    )

    # Create trainer
    trainer = FSDPTrainer(model_config, training_config, fsdp_config)

    # Create dataloader
    total_batch = training_config.batch_size * training_config.gradient_accumulation_steps
    dataloader = create_dummy_dataloader(
        batch_size=total_batch,
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        rank=trainer.rank,
        world_size=trainer.world_size,
    )

    # Training loop
    if trainer.is_main_process:
        print("\n" + "=" * 60)
        print("Starting FSDP training...")
        print("=" * 60 + "\n")

        mem = trainer.get_memory_stats()
        print(f"Initial memory: {mem['allocated_gb']:.2f} GB allocated")

    data_iter = iter(dataloader)
    start_time = time.time()

    for step in range(training_config.max_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {"input_ids": batch[0]}
        metrics = trainer.train_step(batch)

        if step % training_config.log_interval == 0 and trainer.is_main_process:
            elapsed = time.time() - start_time
            tokens_per_sec = metrics["tokens"] / elapsed
            mem = trainer.get_memory_stats()

            print(
                f"Step {step:6d} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"LR: {metrics['lr']:.2e} | "
                f"Tokens/s: {tokens_per_sec:,.0f} | "
                f"Mem: {mem['allocated_gb']:.1f}GB"
            )

        if step > 0 and step % training_config.save_interval == 0:
            trainer.save_checkpoint(f"{training_config.checkpoint_dir}/step_{step}.pt")

    trainer.save_checkpoint(f"{training_config.checkpoint_dir}/final.pt")

    if trainer.is_main_process:
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time:.2f}s")
        print(f"Tokens processed: {trainer.tokens_seen:,}")
        print(f"Peak memory: {trainer.get_memory_stats()['max_allocated_gb']:.2f} GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
