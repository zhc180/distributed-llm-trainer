"""Distributed Data Parallel (DDP) Trainer.

This is the foundation of distributed training. Start here!

Usage:
    # Single GPU
    python ddp_trainer.py --config configs/small_model.yaml

    # Multi-GPU (e.g., 4 GPUs)
    torchrun --nproc_per_node=4 ddp_trainer.py --config configs/small_model.yaml
"""
import argparse
import os
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt import GPT, count_parameters
from models.config import GPTConfig


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Data
    batch_size: int = 8
    max_seq_len: int = 1024

    # Optimization
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # Schedule
    max_steps: int = 10000
    warmup_steps: int = 1000
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000

    # Mixed precision
    mixed_precision: str = "bf16"  # "fp32", "fp16", "bf16"

    # Gradient accumulation
    gradient_accumulation_steps: int = 4

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    resume_from: Optional[str] = None


class DistributedTrainer:
    """DDP Trainer for distributed training across multiple GPUs."""

    def __init__(
        self,
        model_config: GPTConfig,
        training_config: TrainingConfig,
    ):
        self.model_config = model_config
        self.training_config = training_config

        # Initialize distributed environment
        self._setup_distributed()

        # Setup device and dtype
        self._setup_device()

        # Build model
        self._setup_model()

        # Setup optimizer and scheduler
        self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.tokens_seen = 0

    def _setup_distributed(self):
        """Initialize distributed training environment."""
        self.distributed = dist.is_initialized() or "RANK" in os.environ

        if self.distributed:
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")

            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

        self.is_main_process = self.rank == 0

        if self.is_main_process:
            print(f"Distributed training: {self.distributed}")
            print(f"World size: {self.world_size}")

    def _setup_device(self):
        """Setup device and mixed precision context."""
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        # Setup mixed precision
        mp = self.training_config.mixed_precision
        if mp == "bf16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            self.autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif mp == "fp16":
            self.dtype = torch.float16
            self.autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            self.dtype = torch.float32
            self.autocast_ctx = nullcontext()

        # Gradient scaler for fp16
        self.scaler = torch.amp.GradScaler() if mp == "fp16" else None

        if self.is_main_process:
            print(f"Device: {self.device}")
            print(f"Mixed precision: {mp} (dtype: {self.dtype})")

    def _setup_model(self):
        """Build and wrap model with DDP."""
        # Create model
        self.model = GPT(self.model_config).to(self.device)

        if self.is_main_process:
            print(f"Model parameters: {count_parameters(self.model):,}")

        # Wrap with DDP if distributed
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )

    def _setup_optimizer(self):
        """Setup AdamW optimizer with weight decay.

        Not all parameters should get weight decay! This is a common mistake.
        Weight decay is L2 regularization - it pushes weights toward zero.
        But biases and normalization layers shouldn't be regularized.

        Steps:
        1. Create two lists: decay_params and no_decay_params

        2. Loop through self.model.named_parameters():
           - Only include params where param.requires_grad is True
           - NO decay for: bias parameters, layernorm/norm parameters
             (check if "bias" in name or "norm" in name)
           - YES decay for: everything else (weight matrices)

        3. Create parameter groups (a list of dicts):
           [
               {"params": decay_params, "weight_decay": cfg.weight_decay},
               {"params": no_decay_params, "weight_decay": 0.0},
           ]

        4. Create the optimizer:
           self.optimizer = torch.optim.AdamW(
               optim_groups,
               lr=cfg.learning_rate,
               betas=(cfg.beta1, cfg.beta2),
               fused=torch.cuda.is_available(),  # Fused kernel = faster on GPU
           )

        Why separate groups?
        - Weight matrices benefit from regularization (prevents overfitting)
        - Biases are small, regularizing them hurts more than helps
        - Norm layers have a specific scale meaning, decay would break it

        Why AdamW over Adam?
        - AdamW decouples weight decay from gradient updates
        - Adam applies decay to the gradient, which interacts badly with
          adaptive learning rates. AdamW applies it directly to weights.
        """
        cfg = self.training_config
        decay_params, no_decay_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optim_groups = [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            optim_groups,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            fused=torch.cuda.is_available(),
        )
        

    def get_lr(self, step: int) -> float:
        """Cosine learning rate schedule with warmup.

        This is the standard schedule used in most LLM training:

        Phase 1 - Linear Warmup (step < warmup_steps):
            - LR increases linearly from 0 to max_lr
            - Formula: lr = max_lr * (step / warmup_steps)
            - Why? Helps stabilize early training when gradients are noisy

        Phase 2 - Cosine Decay (step >= warmup_steps):
            - LR decreases following a cosine curve to min_lr
            - decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
            - coeff = 0.5 * (1 + cos(pi * decay_ratio))  # goes from 1 to 0
            - lr = min_lr + coeff * (max_lr - min_lr)
            - Use min_lr = 0.1 * max_lr (10% of peak)

        Args:
            step: Current training step

        Returns:
            Learning rate for this step

        Hint: You'll need math.cos and math.pi
        """
        import math
        cfg = self.training_config
        if step < cfg.warmup_steps:
            lr = cfg.learning_rate * (step / cfg.warmup_steps)
        else:
            decay_ratio = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
            coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            min_lr = 0.1 * cfg.learning_rate
            lr = min_lr + coeff * (cfg.learning_rate - min_lr)
        return lr


    def train_step(self, batch: dict) -> dict:
        """Single training step with gradient accumulation.

        Gradient accumulation simulates larger batch sizes by:
        1. Splitting the batch into micro-batches
        2. Running forward+backward on each micro-batch
        3. Accumulating gradients (they sum automatically!)
        4. Only calling optimizer.step() after all micro-batches

        Why? If you want batch_size=32 but only fit batch_size=8 in memory,
        use gradient_accumulation_steps=4.

        Steps to implement:
        1. Set model to train mode, zero gradients
        2. Move input_ids to device, clone as labels
        3. Loop over micro-batches:
           a. Slice the batch: [start_idx:end_idx]
           b. Forward pass inside autocast context (self.autocast_ctx)
           c. IMPORTANT: Divide loss by gradient_accumulation_steps!
           d. Backward pass (handle self.scaler for FP16 if not None)
           e. Accumulate loss for logging
        4. Gradient clipping (use torch.nn.utils.clip_grad_norm_)
           - If using scaler, call scaler.unscale_(optimizer) first!
        5. Optimizer step (handle scaler if not None)
        6. Update learning rate using self.get_lr()
        7. Update self.global_step and self.tokens_seen

        Mixed precision notes:
        - self.autocast_ctx: Context manager for automatic mixed precision
        - self.scaler: GradScaler for FP16 (None for BF16/FP32)
          - scaler.scale(loss).backward() instead of loss.backward()
          - scaler.step(optimizer) instead of optimizer.step()
          - scaler.update() after step

        Returns:
            dict with "loss", "lr", "tokens" keys
        """
        cfg = self.training_config

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        input_ids = batch["input_ids"].to(self.device)
        labels = input_ids.clone()

        micro_bs = input_ids.shape[0] // cfg.gradient_accumulation_steps

        for micro_step in range(cfg.gradient_accumulation_steps):
            start = micro_bs * micro_step
            end = start + micro_bs

            micro_input = input_ids[start:end]
            micro_labels = labels[start:end]

            sync_ctx = (
                self.model.no_sync()
                if self.distributed and micro_step < cfg.gradient_accumulation_steps - 1 else nullcontext()
            )

            with sync_ctx:
                with self.autocast_ctx:
                    _, loss = self.model(micro_input, labels=micro_labels)
                    loss = loss / cfg.gradient_accumulation_steps
                
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

            
            total_loss += loss.item()

        if cfg.grad_clip > 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
    
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)

        lr = self.get_lr(self.global_step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.global_step += 1
        self.tokens_seen += input_ids.numel() * self.world_size

        return {"loss": total_loss, "lr": lr, "tokens": self.tokens_seen}


    def save_checkpoint(self, path: str):
        """Save training checkpoint.

        Checkpointing lets you resume training after interruptions.
        In distributed training, only rank 0 should save (otherwise
        all ranks write the same file simultaneously).

        Steps:
        1. Early return if not self.is_main_process (only rank 0 saves)

        2. Create the directory: os.makedirs(os.path.dirname(path), exist_ok=True)

        3. Get the unwrapped model state dict:
           - DDP wraps your model in a .module attribute
           - If self.distributed: use self.model.module.state_dict()
           - Otherwise: use self.model.state_dict()
           - Why unwrap? So the saved weights don't have "module." prefix
             in every key, making them loadable without DDP

        4. Build checkpoint dict with:
           - "model": the model state dict
           - "optimizer": self.optimizer.state_dict()
           - "global_step": self.global_step
           - "tokens_seen": self.tokens_seen
           - "model_config": self.model_config
           - "training_config": self.training_config

        5. Save with torch.save(checkpoint, path)

        Hint: torch.save uses Python's pickle under the hood
        """
        if not self.is_main_process:
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)

        model = self.model.module if self.distributed else self.model

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "model_config": self.model_config,
            "training_config": self.training_config
        }

        torch.save(checkpoint, path)



    def load_checkpoint(self, path: str):
        """Load training checkpoint.

        Steps:
        1. Load checkpoint dict:
           checkpoint = torch.load(path, map_location=self.device)
           (map_location ensures tensors go to the right device)

        2. Get the unwrapped model (same DDP unwrapping as save):
           model = self.model.module if self.distributed else self.model

        3. Load model weights: model.load_state_dict(checkpoint["model"])

        4. Load optimizer state: self.optimizer.load_state_dict(checkpoint["optimizer"])

        5. Restore training state:
           - self.global_step = checkpoint["global_step"]
           - self.tokens_seen = checkpoint["tokens_seen"]

        6. Print confirmation (only on main process)

        Why load optimizer state?
        - AdamW tracks running averages (momentum, variance) per parameter
        - Without restoring these, training "restarts" with cold optimizer
        - This causes a spike in loss after resuming
        """
        checkpoint = torch.load(path, map_location=self.device)
        model = self.model.module if self.distributed else self.model

        model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.global_step = checkpoint["global_step"]
        self.tokens_seen = checkpoint["tokens_seen"]

        if self.is_main_process():
            print(f"Loaded Checkpoint from {path} (step {self.global_step})")



def create_dummy_dataloader(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    num_batches: int = 1000,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    """Create a dummy dataloader for testing.

    In practice, you'd use a real dataset like OpenWebText or The Pile.
    """
    # Generate random data
    data = torch.randint(0, vocab_size, (num_batches * batch_size, seq_len))

    dataset = torch.utils.data.TensorDataset(data)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if distributed else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True,
        num_workers=4,
    )


def main():
    import math

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    args = parser.parse_args()

    # Model config
    if args.model_size == "small":
        model_config = GPTConfig.gpt2_small()
    elif args.model_size == "medium":
        model_config = GPTConfig.gpt2_medium()
    else:
        model_config = GPTConfig.gpt2_large()

    model_config.gradient_checkpointing = args.gradient_checkpointing

    # Training config
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        mixed_precision=args.mixed_precision,
    )

    # Create trainer
    trainer = DistributedTrainer(model_config, training_config)

    # Create dataloader
    dataloader = create_dummy_dataloader(
        batch_size=training_config.batch_size * training_config.gradient_accumulation_steps,
        seq_len=model_config.max_seq_len,
        vocab_size=model_config.vocab_size,
        distributed=trainer.distributed,
        rank=trainer.rank,
        world_size=trainer.world_size,
    )

    # Training loop
    if trainer.is_main_process:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60 + "\n")

    data_iter = iter(dataloader)
    start_time = time.time()

    for step in range(training_config.max_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {"input_ids": batch[0]}

        # Train step
        metrics = trainer.train_step(batch)

        # Logging
        if step % training_config.log_interval == 0 and trainer.is_main_process:
            elapsed = time.time() - start_time
            tokens_per_sec = metrics["tokens"] / elapsed

            print(
                f"Step {step:6d} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"LR: {metrics['lr']:.2e} | "
                f"Tokens/sec: {tokens_per_sec:,.0f}"
            )

        # Save checkpoint
        if step > 0 and step % training_config.save_interval == 0:
            trainer.save_checkpoint(f"{training_config.checkpoint_dir}/step_{step}.pt")

    # Final save
    trainer.save_checkpoint(f"{training_config.checkpoint_dir}/final.pt")

    # Cleanup
    if trainer.distributed:
        dist.destroy_process_group()

    if trainer.is_main_process:
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time:.2f}s")
        print(f"Total tokens processed: {trainer.tokens_seen:,}")


if __name__ == "__main__":
    main()
