"""Phase 2 & 3: Backward token sampling from an autoregressive language model.

Given a suffix, samples plausible predecessor tokens using Bayes' rule:
    P(v | suffix) ∝ P(suffix | v) × P(v)

where P(suffix | v) comes from a forward pass on [v, s₁, ..., s_T]
and P(v) is the precomputed unigram (stationary) distribution.
The bigram table guides candidate ordering; the LM scores the full suffix.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from backward_sampler.bigram import BigramArtifacts
from backward_sampler.utils import sample_from_log_probs


def compute_log_z(
    model: PreTrainedModel,
    suffix_ids: list[int],
    log_unigram: np.ndarray,
) -> float:
    """Compute the log-probability of a suffix: log P(s₁, ..., s_T).

    Uses the bigram stationary distribution for the first token and
    the LM's own conditionals for the rest:
        log Z = log π(s₁) + Σᵢ₌₂ᵀ log P_LM(sᵢ | s₁, ..., sᵢ₋₁)

    π(s₁) is the correct marginal because Σ_v P(s₁|v)×π(v) = π(s₁)
    by definition of the stationary distribution. The LM conditionals
    for i≥2 approximate the true marginal (which would require summing
    over v) but are accurate when the suffix context dominates.

    Args:
        model: A causal language model in eval mode.
        suffix_ids: Token IDs of the suffix sequence.
        log_unigram: Precomputed log-unigram (stationary) distribution.

    Returns:
        log Z as a float.
    """
    device = next(model.parameters()).device
    input_ids = torch.tensor([suffix_ids], device=device)  # (1, T)

    # P(s₁) from the unigram (stationary) distribution
    log_z = float(log_unigram[suffix_ids[0]])

    if len(suffix_ids) > 1:
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits  # (1, T, V)

        log_probs = F.log_softmax(logits, dim=-1)  # (1, T, V)

        # Accumulate log P(sᵢ | s₁...sᵢ₋₁) for i = 2..T
        # Position i-1 predicts sᵢ; position T-1 predicts beyond suffix (ignored)
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


_CLOZE_PROMPT = '''\
Fill in the blank token in each example.

Text: "ed the basics of the game" → Blank: "learn"
Text: " you doing today?" → Blank: " are"
Text: "'t believe it" → Blank: "don"
Text: ".\\n\\nThe first" → Blank: "\\n"
Text: " is the capital of France." → Blank: " Paris"
Text: "{suffix_text}" → Blank: "'''


def _get_forward_prompt_order(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    suffix_ids: list[int],
) -> np.ndarray:
    """Get candidate predecessor tokens sorted by a cloze-style forward prompt.

    Builds a few-shot prompt that teaches the LM to predict the token
    preceding a given suffix, then returns all vocab tokens sorted by
    the LM's predicted probability (descending).

    Args:
        model: A causal language model in eval mode.
        tokenizer: The model's tokenizer.
        suffix_ids: Token IDs of the current suffix.

    Returns:
        Array of ALL token IDs sorted by descending heuristic probability.
    """
    suffix_text = tokenizer.decode(suffix_ids)
    prompt = _CLOZE_PROMPT.format(suffix_text=suffix_text)
    prompt_ids = tokenizer.encode(prompt)

    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt_ids], device=device)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # (1, seq_len, V)

    # Take logits at the last position — the LM's prediction for the blank
    last_logits = logits[0, -1, :]  # (V,)
    # Sort all tokens by descending probability
    sorted_indices = torch.argsort(last_logits, descending=True)
    return sorted_indices.cpu().numpy()


def _score_candidate_batch(
    model: PreTrainedModel,
    candidate_tokens: np.ndarray,
    suffix_ids: list[int],
    log_unigram: np.ndarray,
) -> tuple[list[float], list[int]]:
    """Score a batch of candidates by running the LM on [v, s₁, ..., s_T].

    For each candidate v, computes:
        log_score = log P(suffix | v) + log P(v)

    Args:
        model: A causal language model in eval mode.
        candidate_tokens: Array of candidate token IDs to evaluate.
        suffix_ids: Token IDs of the current suffix.
        log_unigram: Precomputed log-unigram distribution.

    Returns:
        Tuple of (log_scores, token_ids) for the batch.
    """
    device = next(model.parameters()).device
    suffix_tensor = torch.tensor(suffix_ids, device=device)  # (T,)
    T = len(suffix_ids)
    B = len(candidate_tokens)

    # Build input sequences: [v, s₁, s₂, ..., s_T] for each candidate v
    v_tensor = torch.tensor(candidate_tokens, device=device).unsqueeze(1)  # (B, 1)
    suffix_expanded = suffix_tensor.unsqueeze(0).expand(B, -1)  # (B, T)
    input_ids = torch.cat([v_tensor, suffix_expanded], dim=1)  # (B, T+1)

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits  # (B, T+1, V)

    log_probs = F.log_softmax(logits, dim=-1)  # (B, T+1, V)

    # Vectorized: gather log P(sⱼ | v, s₁, ..., sⱼ₋₁) for all j and all B
    # Position j predicts suffix_ids[j], for j = 0..T-1
    suffix_indices = torch.tensor(suffix_ids, device=device)  # (T,)
    # Gather: for each batch item i and position j, get log_probs[i, j, suffix_ids[j]]
    gathered = log_probs[:, :T, :].gather(
        2, suffix_indices.view(1, T, 1).expand(B, T, 1)
    ).squeeze(2)  # (B, T)
    log_p_suffix_given_v = gathered.sum(dim=1)  # (B,)

    log_scores: list[float] = []
    token_ids: list[int] = []
    for i, v in enumerate(candidate_tokens):
        log_p_v = float(log_unigram[v])
        log_scores.append(log_p_suffix_given_v[i].item() + log_p_v)
        token_ids.append(int(v))

    return log_scores, token_ids


def backward_step(
    model: PreTrainedModel,
    suffix_ids: list[int],
    artifacts: BigramArtifacts,
    threshold: float = 0.9,
    batch_size: int = 256,
    temperature: float = 1.0,
    heuristic: Literal["bigram", "forward-prompt"] = "bigram",
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> int:
    """Sample a single token to prepend to the suffix.

    Implements the core Bayesian backward sampling step:
    1. Compute Z (suffix log-probability) via forward pass
    2. Iterate candidates in heuristic-sorted order
    3. Score each: log P(suffix | v) + log P(v) via forward pass on [v]+suffix
    4. Stop when accumulated mass exceeds threshold × Z
    5. Sample from normalized scores (with temperature)

    Args:
        model: A causal language model in eval mode.
        suffix_ids: Token IDs of the current suffix.
        artifacts: Precomputed bigram artifacts.
        threshold: Nucleus stopping threshold (0 to 1). Higher = more
            candidates evaluated, better coverage.
        batch_size: Number of candidates to score per forward pass.
        temperature: Sampling temperature. <1 sharpens the distribution,
            >1 flattens it. Default 1.0 (no scaling).
        heuristic: Candidate ordering strategy. "bigram" uses the precomputed
            reverse bigram matrix; "forward-prompt" uses a cloze-style LM prompt.
        tokenizer: Required when heuristic="forward-prompt".

    Returns:
        Sampled token ID to prepend to the suffix.
    """

    # Validate parameters early
    if heuristic == "forward-prompt" and tokenizer is None:
        raise ValueError("tokenizer is required for heuristic='forward-prompt'")

    # Step 1: Compute Z = P(suffix)
    log_z = compute_log_z(model, suffix_ids, artifacts.log_unigram)

    # Step 2: Get candidate order from heuristic
    if heuristic == "forward-prompt":
        candidates = _get_forward_prompt_order(model, tokenizer, suffix_ids)
        # DO NOT REMOVE: The heuristic must return ALL tokens. No top-p filtering.
        # Only the Bayesian nucleus stopping criterion may prune candidates.
        assert len(candidates) == tokenizer.vocab_size, (
            f"Forward-prompt heuristic must return all {tokenizer.vocab_size} tokens, "
            f"got {len(candidates)}. Do not apply top-p to the heuristic."
        )
    else:
        s1 = suffix_ids[0]
        candidates = _get_candidate_order(artifacts, s1)

    if len(candidates) == 0:
        # Fallback: if no bigram candidates, sample from unigram
        return sample_from_log_probs(torch.tensor(artifacts.log_unigram))

    # Step 3: Score candidates in batches, stopping when nucleus threshold met
    all_log_scores: list[float] = []
    all_token_ids: list[int] = []
    accumulated_log_mass = float("-inf")  # log(0)

    for batch_start in range(0, len(candidates), batch_size):
        batch_end = min(batch_start + batch_size, len(candidates))
        batch_tokens = candidates[batch_start:batch_end]

        batch_scores, batch_ids = _score_candidate_batch(
            model, batch_tokens, suffix_ids, artifacts.log_unigram,
        )
        all_log_scores.extend(batch_scores)
        all_token_ids.extend(batch_ids)

        # Update accumulated mass: logsumexp over all scores so far
        accumulated_log_mass = torch.logsumexp(
            torch.tensor(all_log_scores), dim=0
        ).item()

        # Check nucleus stopping criterion
        if accumulated_log_mass - log_z > np.log(threshold):
            break

    # Step 4: Sample from normalized distribution (with temperature)
    log_scores_tensor = torch.tensor(all_log_scores) / temperature
    sampled_idx = sample_from_log_probs(log_scores_tensor)

    return all_token_ids[sampled_idx]


def backward_generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    suffix: str,
    artifacts: BigramArtifacts,
    max_new_tokens: int = 50,
    threshold: float = 0.9,
    batch_size: int = 256,
    temperature: float = 1.0,
    verbose: bool = False,
    on_token: Callable[[int, str, int], None] | None = None,
    heuristic: Literal["bigram", "forward-prompt"] = "bigram",
) -> str:
    """Generate tokens backward (prepend) until BOS or max length.

    Args:
        model: A causal language model in eval mode.
        tokenizer: The model's tokenizer.
        suffix: The suffix text to generate backward from.
        artifacts: Precomputed bigram artifacts.
        max_new_tokens: Maximum number of tokens to prepend.
        threshold: Nucleus stopping threshold for each backward step.
        batch_size: Number of candidates to score per forward pass.
        temperature: Sampling temperature. <1 sharpens, >1 flattens.
        verbose: If True, print each generated token.
        on_token: Optional callback invoked after each token is sampled.
            Called with (token_id, decoded_text, step_number).
        heuristic: Candidate ordering strategy. "bigram" uses the precomputed
            reverse bigram matrix; "forward-prompt" uses a cloze-style LM prompt.

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
            batch_size=batch_size,
            temperature=temperature,
            heuristic=heuristic,
            tokenizer=tokenizer,
        )

        # GPT-2 uses the same token (50256) for BOS and EOS
        if token == eos_token_id:
            if verbose:
                print(f"[Step {step + 1}] Sampled BOS/EOS — stopping.")
            if on_token is not None:
                on_token(token, "<|endoftext|>", step + 1)
            break

        suffix_ids = [token] + suffix_ids
        decoded = tokenizer.decode([token])

        if verbose:
            print(f"[Step {step + 1}] Sampled: {decoded!r}")

        if on_token is not None:
            on_token(token, decoded, step + 1)

    return tokenizer.decode(suffix_ids)
