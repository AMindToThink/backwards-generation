"""Phase 1: Bigram precomputation from a language model.

Passes every token in the vocabulary through the model individually,
recording the output distribution to build a bigram transition matrix.
This matrix is then used to derive a unigram (stationary) distribution
and to efficiently order candidate tokens during backward sampling.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from tqdm import trange
from transformers import PreTrainedModel


@dataclasses.dataclass
class BigramArtifacts:
    """Precomputed bigram statistics extracted from a language model.

    Attributes:
        log_bigram_csr: Sparse CSR matrix of shape (V, V) containing
            log P(w | v) — the log-probability of token w following token v.
        log_bigram_csc: Same matrix in CSC format for efficient column access
            (used to look up which tokens v are likely predecessors of a given w).
        log_unigram: Dense array of shape (V,) containing log P(v) — the
            stationary distribution of the bigram Markov chain.
    """

    log_bigram_csr: sp.csr_matrix
    log_bigram_csc: sp.csc_matrix
    log_unigram: np.ndarray

    def save(self, directory: str | Path) -> None:
        """Save artifacts to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        sp.save_npz(directory / "log_bigram.npz", self.log_bigram_csr)
        np.save(directory / "log_unigram.npy", self.log_unigram)

    @classmethod
    def load(cls, directory: str | Path) -> BigramArtifacts:
        """Load artifacts from a directory."""
        directory = Path(directory)
        log_bigram_csr = sp.load_npz(directory / "log_bigram.npz").tocsr()
        log_bigram_csc = log_bigram_csr.tocsc()
        log_unigram = np.load(directory / "log_unigram.npy")
        return cls(
            log_bigram_csr=log_bigram_csr,
            log_bigram_csc=log_bigram_csc,
            log_unigram=log_unigram,
        )


def _apply_top_p(log_probs: np.ndarray, top_p: float) -> np.ndarray:
    """Zero out tokens outside the top-p nucleus for each row.

    Args:
        log_probs: Array of shape (B, V) with log-probabilities.
        top_p: Cumulative probability threshold (e.g., 0.95).

    Returns:
        Array of same shape with entries outside the nucleus set to 0.0.
    """
    probs = np.exp(log_probs)
    # Sort descending along vocab axis
    sorted_indices = np.argsort(-probs, axis=-1)
    sorted_probs = np.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative = np.cumsum(sorted_probs, axis=-1)

    # Build mask: keep tokens where cumulative prob <= top_p (plus the first one over)
    # Shift cumulative right by 1 so we include the token that crosses the threshold
    shifted_cumulative = np.zeros_like(cumulative)
    shifted_cumulative[:, 1:] = cumulative[:, :-1]
    cutoff_mask = shifted_cumulative >= top_p

    # Zero out entries beyond the nucleus in the sorted order
    sorted_probs[cutoff_mask] = 0.0

    # Scatter back to original order
    result = np.zeros_like(probs)
    np.put_along_axis(result, sorted_indices, sorted_probs, axis=-1)

    # Convert back to log-space, using 0.0 for zeroed entries (sparse sentinel)
    with np.errstate(divide="ignore"):
        result_log = np.where(result > 0, np.log(result), 0.0)

    return result_log.astype(np.float32)


def compute_bigram_matrix(
    model: PreTrainedModel,
    vocab_size: int,
    batch_size: int = 256,
    top_p: float = 0.95,
) -> sp.csr_matrix:
    """Compute the bigram transition matrix from a language model.

    For each token v in the vocabulary, performs a forward pass to obtain
    log P(w | v) for all w. Only tokens within the top-p nucleus are kept;
    the rest are zeroed out for sparse storage.

    This assumes the forward model will be sampled with top-p nucleus
    sampling at the same threshold, which is standard practice.

    Args:
        model: A causal language model in eval mode.
        vocab_size: Size of the token vocabulary.
        batch_size: Number of tokens to process in each forward pass.
        top_p: Nucleus sampling threshold. Only the smallest set of tokens
            whose cumulative probability exceeds top_p are retained per row.

    Returns:
        Sparse CSR matrix of shape (vocab_size, vocab_size) containing
        log-probabilities.
    """
    device = next(model.parameters()).device

    rows: list[sp.csr_matrix] = []

    for start in trange(0, vocab_size, batch_size, desc="Computing bigram matrix"):
        end = min(start + batch_size, vocab_size)
        token_ids = torch.arange(start, end, device=device).unsqueeze(1)  # (B, 1)

        with torch.no_grad():
            logits = model(input_ids=token_ids).logits  # (B, 1, V)

        log_probs = F.log_softmax(logits[:, 0, :], dim=-1).cpu().numpy()  # (B, model_V)

        # Truncate to vocab_size columns (relevant when computing for a subset)
        log_probs = log_probs[:, :vocab_size]

        # Apply top-p nucleus filtering
        log_probs = _apply_top_p(log_probs, top_p)

        # Convert batch to sparse rows
        batch_sparse = sp.csr_matrix(log_probs)
        rows.append(batch_sparse)

    # Stack all rows into a single sparse matrix
    bigram_matrix = sp.vstack(rows, format="csr")
    return bigram_matrix


def compute_unigram_distribution(
    bigram_matrix: sp.csr_matrix,
    n_iterations: int = 100,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Compute the stationary distribution of the bigram Markov chain.

    Uses power iteration: starting from a uniform distribution, repeatedly
    multiplies by the bigram transition matrix until convergence.

    The bigram matrix contains log-probabilities with zeros for thresholded
    entries. We need to work in probability space for power iteration, so
    we convert, run the iteration, then return the result in log-space.

    Args:
        bigram_matrix: Sparse CSR matrix of shape (V, V) with log-probs.
            Zero entries represent thresholded (near-zero probability) values.
        n_iterations: Maximum number of power iteration steps.
        tolerance: Stop early if L1 change between iterations < tolerance.

    Returns:
        Dense array of shape (V,) containing log P(v) for each token.
    """
    vocab_size = bigram_matrix.shape[0]

    # Convert from log-probs to probabilities for power iteration.
    # The matrix has 0.0 in place of thresholded entries.
    # Non-zero entries are log-probs (negative values).
    # We need to: exp(non-zero entries), leave zero entries as zero.
    prob_matrix = bigram_matrix.copy()
    prob_matrix.data = np.exp(prob_matrix.data)

    # Normalize rows to sum to 1 (they may not due to thresholding)
    row_sums = np.array(prob_matrix.sum(axis=1)).flatten()
    row_sums[row_sums == 0] = 1.0  # avoid division by zero
    normalizer = sp.diags(1.0 / row_sums)
    prob_matrix = normalizer @ prob_matrix

    # Power iteration: p_{n+1} = p_n @ prob_matrix (left eigenvector)
    # Equivalently: p_{n+1} = prob_matrix.T @ p_n
    p = np.ones(vocab_size, dtype=np.float64) / vocab_size

    pbar = trange(n_iterations, desc="Power iteration")
    for i in pbar:
        p_new = prob_matrix.T @ p
        p_new = p_new / p_new.sum()  # normalize

        delta = np.abs(p_new - p).sum()
        p = p_new
        pbar.set_postfix(delta=f"{delta:.2e}")

        if delta < tolerance:
            pbar.set_description("Power iteration (converged)")
            break

    # Convert to log-space
    # Clamp to avoid log(0)
    p = np.maximum(p, 1e-30)
    log_unigram = np.log(p).astype(np.float32)

    return log_unigram


def precompute_artifacts(
    model: PreTrainedModel,
    vocab_size: int,
    batch_size: int = 256,
    top_p: float = 0.95,
    power_iterations: int = 100,
) -> BigramArtifacts:
    """Run the full bigram precomputation pipeline.

    Args:
        model: A causal language model in eval mode.
        vocab_size: Size of the token vocabulary.
        batch_size: Batch size for forward passes.
        top_p: Nucleus sampling threshold for sparse storage.
        power_iterations: Number of power iteration steps for unigram.

    Returns:
        BigramArtifacts containing the bigram matrix and unigram distribution.
    """
    log_bigram_csr = compute_bigram_matrix(
        model, vocab_size, batch_size, top_p
    )
    log_unigram = compute_unigram_distribution(log_bigram_csr, power_iterations)
    log_bigram_csc = log_bigram_csr.tocsc()

    return BigramArtifacts(
        log_bigram_csr=log_bigram_csr,
        log_bigram_csc=log_bigram_csc,
        log_unigram=log_unigram,
    )
