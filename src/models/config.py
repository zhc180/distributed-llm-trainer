"""Model configuration dataclass."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPTConfig:
    """Configuration for GPT model.

    Default values are for GPT-2 124M (small).
    """
    # Model architecture
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: Optional[int] = None  # Defaults to 4 * hidden_size
    max_seq_len: int = 1024

    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1

    # Initialization
    initializer_range: float = 0.02

    # Activation
    activation: str = "gelu"

    # Optimization flags
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size

        assert self.hidden_size % self.num_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_heads})"

    @classmethod
    def gpt2_small(cls) -> "GPTConfig":
        """GPT-2 124M configuration."""
        return cls(
            vocab_size=50257,
            hidden_size=768,
            num_layers=12,
            num_heads=12,
        )

    @classmethod
    def gpt2_medium(cls) -> "GPTConfig":
        """GPT-2 355M configuration."""
        return cls(
            vocab_size=50257,
            hidden_size=1024,
            num_layers=24,
            num_heads=16,
        )

    @classmethod
    def gpt2_large(cls) -> "GPTConfig":
        """GPT-2 774M configuration."""
        return cls(
            vocab_size=50257,
            hidden_size=1280,
            num_layers=36,
            num_heads=20,
        )

    @classmethod
    def gpt2_xl(cls) -> "GPTConfig":
        """GPT-2 1.5B configuration."""
        return cls(
            vocab_size=50257,
            hidden_size=1600,
            num_layers=48,
            num_heads=25,
        )

    def num_parameters(self) -> int:
        """Estimate number of parameters (excluding embeddings)."""
        # Embedding: vocab_size * hidden_size
        embed_params = self.vocab_size * self.hidden_size

        # Position embedding: max_seq_len * hidden_size
        pos_params = self.max_seq_len * self.hidden_size

        # Per transformer layer:
        # - Attention: 4 * hidden_size^2 (Q, K, V, O projections)
        # - MLP: 2 * hidden_size * intermediate_size
        # - LayerNorms: 4 * hidden_size
        attn_params = 4 * self.hidden_size ** 2
        mlp_params = 2 * self.hidden_size * self.intermediate_size
        ln_params = 4 * self.hidden_size
        layer_params = attn_params + mlp_params + ln_params

        # Final layer norm
        final_ln = 2 * self.hidden_size

        total = embed_params + pos_params + (self.num_layers * layer_params) + final_ln
        return total
