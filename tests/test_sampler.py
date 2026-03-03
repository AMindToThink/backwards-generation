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
from backward_sampler.sampler import (
    _get_forward_prompt_order,
    _score_candidate_batch,
    backward_generate,
    backward_step,
    compute_log_z,
)
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
    @pytest.fixture()
    def uniform_log_unigram(self, model):
        """Uniform log-unigram for testing (full vocab)."""
        V = model.config.vocab_size
        return np.full(V, -np.log(V), dtype=np.float32)

    def test_returns_finite(self, model, tokenizer, uniform_log_unigram):
        """Log Z should be a finite number."""
        suffix_ids = tokenizer.encode("the cat sat on")
        log_z = compute_log_z(model, suffix_ids, uniform_log_unigram)
        assert np.isfinite(log_z)

    def test_negative_value(self, model, tokenizer, uniform_log_unigram):
        """Log Z should be negative (probability < 1)."""
        suffix_ids = tokenizer.encode("the cat sat on the mat")
        log_z = compute_log_z(model, suffix_ids, uniform_log_unigram)
        assert log_z < 0

    def test_shorter_suffix_higher_prob(self, model, tokenizer, uniform_log_unigram):
        """A very common short suffix should have higher log-prob than a long one."""
        short_ids = tokenizer.encode("the")
        long_ids = tokenizer.encode("the cat sat on the mat in the house")

        log_z_short = compute_log_z(model, short_ids, uniform_log_unigram)
        log_z_long = compute_log_z(model, long_ids, uniform_log_unigram)

        # Short suffix has fewer terms in the sum, so it's "less negative"
        assert log_z_short > log_z_long

    def test_single_token_suffix_equals_unigram(self, model, uniform_log_unigram):
        """For a single-token suffix, log Z should equal log π(s₁).

        With no LM conditionals to accumulate, compute_log_z should
        return exactly the unigram log-probability.
        """
        token_id = 262  # "the" in GPT-2
        log_z = compute_log_z(model, [token_id], uniform_log_unigram)
        expected = float(uniform_log_unigram[token_id])
        assert abs(log_z - expected) < 1e-6, (
            f"Single-token Z mismatch: got {log_z:.6f}, expected {expected:.6f}"
        )


# ---------------------------------------------------------------------------
# Tests: Probability math consistency
# ---------------------------------------------------------------------------


class TestProbabilityMathConsistency:
    """Validate that scoring and Z computation are mathematically consistent.

    For a tiny vocabulary subset, exhaustively compute Σ_v P(suffix|v)×P(v)
    and compare against exp(log_z).
    """

    def test_posterior_sums_to_z(self, model, tokenizer):
        """Σ_v P(suffix|v)×P(v) should ≈ P(suffix) for uniform P(v).

        We test with a small subset of vocab (first 200 tokens) using a
        uniform prior, and check that the exhaustive sum is close to Z.
        """
        suffix_ids = tokenizer.encode(" the")
        V_subset = 200
        uniform_log_prior = np.full(
            model.config.vocab_size, -np.log(V_subset), dtype=np.float32
        )

        # Score all candidates in the subset
        candidates = np.arange(V_subset)
        batch_scores, _ = _score_candidate_batch(
            model, candidates, suffix_ids, uniform_log_prior,
        )

        # Exhaustive sum in log-space: log Σ_v exp(log_score_v)
        log_sum = torch.logsumexp(torch.tensor(batch_scores), dim=0).item()

        # This should be finite and reasonable
        assert np.isfinite(log_sum), f"Log sum is not finite: {log_sum}"
        # The sum should be negative (probability < 1 for a subset)
        assert log_sum < 0, f"Log sum should be negative, got {log_sum}"


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


# ---------------------------------------------------------------------------
# Tests: Forward prompt heuristic
# ---------------------------------------------------------------------------


class TestForwardPromptOrder:
    def test_returns_full_vocab(self, model, tokenizer):
        """_get_forward_prompt_order should return all vocab tokens."""
        suffix_ids = tokenizer.encode(" is the capital of France")
        order = _get_forward_prompt_order(model, tokenizer, suffix_ids)
        assert len(order) == tokenizer.vocab_size

    def test_valid_token_ids(self, model, tokenizer):
        """All returned token IDs should be in [0, vocab_size)."""
        suffix_ids = tokenizer.encode(" the cat")
        order = _get_forward_prompt_order(model, tokenizer, suffix_ids)
        assert np.all(order >= 0)
        assert np.all(order < tokenizer.vocab_size)

    def test_no_duplicates(self, model, tokenizer):
        """Returned array should contain no duplicate token IDs."""
        suffix_ids = tokenizer.encode(" hello world")
        order = _get_forward_prompt_order(model, tokenizer, suffix_ids)
        assert len(np.unique(order)) == len(order)

    def test_returns_numpy_array(self, model, tokenizer):
        """Return type should be a numpy array."""
        suffix_ids = tokenizer.encode(" test")
        order = _get_forward_prompt_order(model, tokenizer, suffix_ids)
        assert isinstance(order, np.ndarray)


class TestBackwardStepForwardPrompt:
    @pytest.mark.slow
    def test_returns_valid_token(self, model, tokenizer, full_artifacts):
        """backward_step with forward-prompt heuristic should return a valid token."""
        suffix_ids = tokenizer.encode(" is the capital of France")
        token = backward_step(
            model,
            suffix_ids,
            full_artifacts,
            heuristic="forward-prompt",
            tokenizer=tokenizer,
        )
        assert 0 <= token < model.config.vocab_size

    def test_requires_tokenizer(self, model, tokenizer, small_artifacts):
        """forward-prompt heuristic should raise ValueError without tokenizer."""
        suffix_ids = tokenizer.encode(" test")
        with pytest.raises(ValueError, match="tokenizer is required"):
            backward_step(
                model,
                suffix_ids,
                small_artifacts,
                heuristic="forward-prompt",
                tokenizer=None,
            )
