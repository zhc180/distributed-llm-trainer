#!/usr/bin/env python3
"""Simple inference script for a trained GPT checkpoint."""
import argparse
import os
import sys

import torch
from transformers import GPT2TokenizerFast

# Add repo root and src to path for imports (needed for checkpoint unpickling)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, REPO_ROOT)

from models.config import GPTConfig
from models.gpt import GPT


class TrainingConfig:
    """Placeholder to satisfy checkpoints saved from ddp_trainer.py."""
    pass


def _get_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="small", choices=["small", "medium", "large"])
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.model_size == "small":
        model_config = GPTConfig.gpt2_small()
    elif args.model_size == "medium":
        model_config = GPTConfig.gpt2_medium()
    else:
        model_config = GPTConfig.gpt2_large()

    device = _get_device(args.device)

    model = GPT(model_config).to(device)
    try:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)

    print(tokenizer.decode(output[0]))


if __name__ == "__main__":
    main()
