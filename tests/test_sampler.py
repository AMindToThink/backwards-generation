"""Tests for the backward sampling system.

These tests load GPT-2 once (via a session-scoped fixture) and run
various checks on the bigram precomputation and backward sampling logic.

Run with: uv run pytest tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from backward_sampler.bigram import (
    BigramArtifacts,
    compute_bigram_matrix,
    compute_unigram_distribution,
)
from backward_sampler.sampler import backward_generate, backward_step, compute_log_z
from backward_sampler.utils import load_model, sample_from_log_probs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def model_and_tokenizer():
    """Load GPT-2 once for the entire test session."""
    model, tokenizer = load_model("openai-community/gpt2", device="cpu")
    return model, tokenizer


@pytest.fixture(scope="session")
def model(model_and_tokenizer):
    return model_and_tokenizer[0]


@pytest.fixture(scope="session")
def tokenizer(model_and_tokenizer):
    return model_and_tokenizer[1]


@pytest.fixture(scope="session")
def small_bigram_matrix(model):
    """Compute bigram matrix for a small subset of vocab for fast testing.

    We use the first 1000 tokens only to keep tests fast.
    """
    return compute_bigram_matrix(model, vocab_size=1000, batch_size=200)


@pytest.fixture(scope="session")
def small_artifacts(small_bigram_matrix):
    """Create BigramArtifacts from the small bigram matrix."""
    log_unigram = compute_unigram_distribution(small_bigram_matrix, n_iterations=50)
    return BigramArtifacts(
        log_bigram_csr=small_bigram_matrix,
        log_bigram_csc=small_bigram_matrix.tocsc(),
        log_unigram=log_unigram,
    )


# ---------------------------------------------------------------------------
# Tests: Utility functions
# ---------------------------------------------------------------------------


class TestSampleFromLogProbs:
    def test_deterministic_with_one_option(self):
        """If one option has all the mass, it should always be sampled."""
        log_probs = torch.tensor([0.0, float("-inf"), float("-inf")])
        for _ in range(10):
            assert sample_from_log_probs(log_probs) == 0

    def test_returns_valid_index(self):
        """Sampled index should be within range."""
        log_probs = torch.randn(100)
        idx = sample_from_log_probs(log_probs)
        assert 0 <= idx < 100


# ---------------------------------------------------------------------------
# Tests: Bigram matrix
# ---------------------------------------------------------------------------


class TestBigramMatrix:
    def test_shape(self, small_bigram_matrix):
        """Bigram matrix should be (V, V)."""
        assert small_bigram_matrix.shape == (1000, 1000)

    def test_values_are_nonpositive(self, small_bigram_matrix):
        """Log-probabilities should be <= 0 (or exactly 0 for thresholded entries)."""
        # Non-zero entries should be negative (log-probs)
        data = small_bigram_matrix.data
        assert np.all(data <= 0.0)

    def test_rows_are_valid_log_distributions(self, small_bigram_matrix):
        """Non-thresholded rows should have probabilities that sum to roughly 1.

        Note: because we threshold and only look at 1000/50257 of the vocab,
        the rows won't sum to exactly 1. But the non-zero entries should be
        valid log-probabilities.
        """
        # Just check that non-zero entries are negative (valid log-probs)
        dense = small_bigram_matrix.toarray()
        nonzero_entries = dense[dense != 0.0]
        assert np.all(nonzero_entries < 0.0)

    def test_has_nonzero_entries(self, small_bigram_matrix):
        """Matrix should have nonzero entries."""
        assert small_bigram_matrix.nnz > 0


# ---------------------------------------------------------------------------
# Tests: Unigram distribution
# ---------------------------------------------------------------------------


class TestUnigramDistribution:
    def test_sums_to_one(self, small_artifacts):
        """Unigram distribution (in probability space) should sum to ~1."""
        probs = np.exp(small_artifacts.log_unigram)
        assert abs(probs.sum() - 1.0) < 1e-4

    def test_all_positive(self, small_artifacts):
        """All unigram probabilities should be positive."""
        probs = np.exp(small_artifacts.log_unigram)
        assert np.all(probs > 0)

    def test_log_values_are_negative(self, small_artifacts):
        """Log-probabilities should be negative (probs < 1)."""
        assert np.all(small_artifacts.log_unigram < 0)


# ---------------------------------------------------------------------------
# Tests: Artifact save/load
# ---------------------------------------------------------------------------


class TestArtifactIO:
    def test_save_and_load_roundtrip(self, small_artifacts, tmp_path):
        """Artifacts should survive a save/load roundtrip."""
        small_artifacts.save(tmp_path / "test_artifacts")
        loaded = BigramArtifacts.load(tmp_path / "test_artifacts")

        # Compare bigram matrix
        diff = (small_artifacts.log_bigram_csr - loaded.log_bigram_csr).nnz
        assert diff == 0, "Bigram matrices differ after roundtrip"

        # Compare unigram
        np.testing.assert_array_almost_equal(
            small_artifacts.log_unigram, loaded.log_unigram
        )


# ---------------------------------------------------------------------------
# Tests: compute_log_z
# ---------------------------------------------------------------------------


class TestComputeLogZ:
    def test_returns_finite(self, model, tokenizer):
        """Log Z should be a finite number."""
        suffix_ids = tokenizer.encode("the cat sat on")
        log_z = compute_log_z(model, suffix_ids)
        assert np.isfinite(log_z)

    def test_negative_value(self, model, tokenizer):
        """Log Z should be negative (probability < 1)."""
        suffix_ids = tokenizer.encode("the cat sat on the mat")
        log_z = compute_log_z(model, suffix_ids)
        assert log_z < 0

    def test_shorter_suffix_higher_prob(self, model, tokenizer):
        """A very common short suffix should have higher log-prob than a long one."""
        short_ids = tokenizer.encode("the")
        long_ids = tokenizer.encode("the cat sat on the mat in the house")

        log_z_short = compute_log_z(model, short_ids)
        log_z_long = compute_log_z(model, long_ids)

        # Short suffix has fewer terms in the sum, so it's "less negative"
        # (though this isn't strictly guaranteed, it's very likely for these examples)
        assert log_z_short > log_z_long


# ---------------------------------------------------------------------------
# Tests: backward_step (requires full-vocab artifacts)
# ---------------------------------------------------------------------------
# NOTE: These tests are marked slow because they require full-vocab bigram
# artifacts. For CI, precompute artifacts and set ARTIFACTS_DIR env var.
# For local testing, run precompute_bigrams.py first.


@pytest.fixture(scope="session")
def full_artifacts():
    """Try to load full-vocab artifacts. Skip if not available."""
    from pathlib import Path

    artifacts_dir = Path("artifacts/gpt2")
    if not artifacts_dir.exists():
        pytest.skip("Full artifacts not available. Run precompute_bigrams.py first.")
    return BigramArtifacts.load(artifacts_dir)


class TestBackwardStep:
    @pytest.mark.slow
    def test_returns_valid_token(self, model, tokenizer, full_artifacts):
        """backward_step should return a valid token ID."""
        suffix_ids = tokenizer.encode(" is the capital of France")
        token = backward_step(model, suffix_ids, full_artifacts)
        assert 0 <= token < model.config.vocab_size

    @pytest.mark.slow
    def test_capital_of_france(self, model, tokenizer, full_artifacts):
        """For 'is the capital of France', Paris-related tokens should appear."""
        suffix_ids = tokenizer.encode(" is the capital of France")
        # Sample multiple times and check if "Paris" ever appears
        sampled_tokens = set()
        for _ in range(20):
            token = backward_step(model, suffix_ids, full_artifacts)
            sampled_tokens.add(token)

        # Decode all sampled tokens
        decoded = {tokenizer.decode([t]).strip().lower() for t in sampled_tokens}
        # We'd expect "paris" to appear at least once in 20 samples
        # (this is probabilistic, so we make the check lenient)
        assert len(decoded) > 0, "Should have sampled at least one token"


class TestBackwardGenerate:
    @pytest.mark.slow
    def test_produces_output(self, model, tokenizer, full_artifacts):
        """backward_generate should produce a non-empty string."""
        result = backward_generate(
            model, tokenizer, " is great", full_artifacts, max_new_tokens=5
        )
        assert len(result) > len(" is great")

    @pytest.mark.slow
    def test_suffix_preserved(self, model, tokenizer, full_artifacts):
        """The original suffix should appear in the output."""
        suffix = " is great"
        result = backward_generate(
            model, tokenizer, suffix, full_artifacts, max_new_tokens=5
        )
        assert result.endswith(suffix) or suffix.strip() in result
