"""Shared utilities for backward sampling."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


def load_model(
    model_name: str = "openai-community/gpt2",
    device: str = "cpu",
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal language model and its tokenizer.

    Args:
        model_name: HuggingFace model identifier.
        device: Device to load the model on ("cpu", "cuda", etc.).

    Returns:
        Tuple of (model, tokenizer).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, tokenizer


def sample_from_log_probs(log_probs: torch.Tensor) -> int:
    """Sample a single index from a log-probability distribution.

    Args:
        log_probs: 1-D tensor of log-probabilities (not necessarily normalized;
            will be normalized via log-softmax before sampling).

    Returns:
        Sampled index as a Python int.
    """
    normalized = F.log_softmax(log_probs, dim=-1)
    probs = normalized.exp()
    return torch.multinomial(probs, num_samples=1).item()
