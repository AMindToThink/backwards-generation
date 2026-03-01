"""Phase 2 & 3: Backward token sampling from an autoregressive language model.

Given a suffix, samples plausible predecessor tokens using Bayes' rule:
    P(v | suffix) ∝ P(s₁ | v) × P(v)

where P(s₁ | v) comes from a forward pass and P(v) is the precomputed
unigram (stationary) distribution.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from backward_sampler.bigram import BigramArtifacts
from backward_sampler.utils import sample_from_log_probs


def compute_log_z(
    model: PreTrainedModel,
    suffix_ids: list[int],
) -> float:
    """Compute the log-probability of a suffix under the forward model.

    Performs a single forward pass on the suffix and accumulates:
        log Z = Σᵢ log P(sᵢ | s₁, ..., sᵢ₋₁)

    For the first token s₁, there is no conditioning context, so we skip it
    (it has no "given" prefix within the suffix itself).

    Args:
        model: A causal language model in eval mode.
        suffix_ids: Token IDs of the suffix sequence.

    Returns:
        log Z as a float.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([suffix_ids], device=device)  # (1, T)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # (1, T, V)

    log_probs = F.log_softmax(logits, dim=-1)  # (1, T, V)

    # Accumulate log P(sᵢ | s₁...sᵢ₋₁) for i = 2..T
    # logits at position i-1 predict token at position i
    log_z = 0.0
    for i in range(1, len(suffix_ids)):
        log_z += log_probs[0, i - 1, suffix_ids[i]].item()

    return log_z


def _get_candidate_order(
    artifacts: BigramArtifacts,
    first_suffix_token: int,
) -> np.ndarray:
    """Get candidate predecessor tokens sorted by reverse bigram probability.

    Extracts column `first_suffix_token` from the CSC bigram matrix and
    returns token indices sorted by descending log-probability.

    Args:
        artifacts: Precomputed bigram artifacts.
        first_suffix_token: The first token of the current suffix (s₁).

    Returns:
        Array of token IDs sorted by P(v | first_suffix_token) descending,
        containing only tokens with nonzero bigram probability.
    """
    col = artifacts.log_bigram_csc.getcol(first_suffix_token)
    col_dense = col.toarray().flatten()  # (V,)

    # Get indices of nonzero entries (these are the plausible predecessors)
    nonzero_mask = col_dense != 0.0
    nonzero_indices = np.where(nonzero_mask)[0]
    nonzero_values = col_dense[nonzero_indices]

    # Sort by descending log-probability
    sorted_order = np.argsort(-nonzero_values)
    return nonzero_indices[sorted_order]


def backward_step(
    model: PreTrainedModel,
    suffix_ids: list[int],
    artifacts: BigramArtifacts,
    threshold: float = 0.9,
    candidate_batch_size: int = 256,
) -> int:
    """Sample a single token to prepend to the suffix.

    Implements the core Bayesian backward sampling step:
    1. Compute Z (suffix log-probability) via forward pass
    2. Iterate candidates in bigram-sorted order
    3. Score each: log P(s₁ | v) + log P(v)
    4. Stop when accumulated mass exceeds threshold × Z
    5. Sample from normalized scores

    Args:
        model: A causal language model in eval mode.
        suffix_ids: Token IDs of the current suffix.
        artifacts: Precomputed bigram artifacts.
        threshold: Nucleus stopping threshold (0 to 1). Higher = more
            candidates evaluated, better coverage.
        candidate_batch_size: Number of candidates to score per forward pass.

    Returns:
        Sampled token ID to prepend to the suffix.
    """
    device = next(model.parameters()).device

    # Step 1: Compute Z
    log_z = compute_log_z(model, suffix_ids)

    # Step 2: Get candidate order from reverse bigram
    s1 = suffix_ids[0]
    candidates = _get_candidate_order(artifacts, s1)

    if len(candidates) == 0:
        # Fallback: if no bigram candidates, sample from unigram
        return sample_from_log_probs(torch.tensor(artifacts.log_unigram))

    # Step 3: Score candidates in batches
    all_log_scores: list[float] = []
    all_token_ids: list[int] = []
    accumulated_log_mass = float("-inf")  # log(0)

    for batch_start in range(0, len(candidates), candidate_batch_size):
        batch_end = min(batch_start + candidate_batch_size, len(candidates))
        batch_tokens = candidates[batch_start:batch_end]

        # Forward pass: each candidate is a single token
        input_ids = torch.tensor(batch_tokens, device=device).unsqueeze(1)  # (B, 1)

        with torch.no_grad():
            logits = model(input_ids=input_ids).logits  # (B, 1, V)

        log_probs = F.log_softmax(logits[:, 0, :], dim=-1)  # (B, V)

        for i, v in enumerate(batch_tokens):
            # log P(s₁ | v) from the forward model
            log_p_s1_given_v = log_probs[i, s1].item()
            # log P(v) from the precomputed unigram
            log_p_v = float(artifacts.log_unigram[v])

            log_score = log_p_s1_given_v + log_p_v
            all_log_scores.append(log_score)
            all_token_ids.append(int(v))

        # Update accumulated mass: logsumexp over all scores so far
        accumulated_log_mass = torch.logsumexp(
            torch.tensor(all_log_scores), dim=0
        ).item()

        # Check nucleus stopping criterion
        if accumulated_log_mass - log_z > np.log(threshold):
            break

    # Step 4: Sample from normalized distribution
    log_scores_tensor = torch.tensor(all_log_scores)
    sampled_idx = sample_from_log_probs(log_scores_tensor)

    return all_token_ids[sampled_idx]


def backward_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    suffix: str,
    artifacts: BigramArtifacts,
    max_new_tokens: int = 50,
    threshold: float = 0.9,
    candidate_batch_size: int = 256,
    verbose: bool = False,
) -> str:
    """Generate tokens backward (prepend) until BOS or max length.

    Args:
        model: A causal language model in eval mode.
        tokenizer: The model's tokenizer.
        suffix: The suffix text to generate backward from.
        artifacts: Precomputed bigram artifacts.
        max_new_tokens: Maximum number of tokens to prepend.
        threshold: Nucleus stopping threshold for each backward step.
        candidate_batch_size: Batch size for candidate scoring.
        verbose: If True, print each generated token.

    Returns:
        The full generated sequence (prefix + suffix) as a string.
    """
    suffix_ids = tokenizer.encode(suffix)
    eos_token_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        token = backward_step(
            model,
            suffix_ids,
            artifacts,
            threshold=threshold,
            candidate_batch_size=candidate_batch_size,
        )

        # GPT-2 uses the same token (50256) for BOS and EOS
        if token == eos_token_id:
            if verbose:
                print(f"[Step {step + 1}] Sampled BOS/EOS — stopping.")
            break

        suffix_ids = [token] + suffix_ids

        if verbose:
            decoded = tokenizer.decode([token])
            print(f"[Step {step + 1}] Sampled: {decoded!r}")

    return tokenizer.decode(suffix_ids)
