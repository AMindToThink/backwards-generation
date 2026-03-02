# Backward Sampling — Progress Report

## What's Implemented

### Phase 1: Bigram Precomputation (`bigram.py`)
- One-time forward pass of all 50,257 GPT-2 tokens to build bigram transition matrix
- Top-p=0.95 nucleus sparsification (35% density, ~887M nonzeros)
- Unigram (stationary) distribution via power iteration on bigram chain
- Save/load artifacts to disk (~2-3 GB)
- Precomputation takes ~7 min on GPU (cuda:0, batch_size=512)

### Phase 2: Single-Step Backward Sampler (`sampler.py`)
- Scores candidates using full suffix: `log P(suffix | v) + log P(v)`
- Forward pass on `[v, s₁, ..., s_T]` for each candidate batch
- Bigram table orders candidates (which to evaluate first)
- Nucleus stopping criterion compares accumulated mass to Z = P(suffix)
- Temperature parameter for controlling sampling sharpness

### Phase 3: Multi-Step Generation (`sampler.py` + `scripts/backward_generate.py`)
- Iterative prepending loop with BOS stopping
- Streaming output (--stream flag)
- CLI with full configurability: --batch-size, --threshold, --temperature, etc.

## Known Issues

### 1. Nucleus stopping criterion never triggers
The accumulated candidate mass converges to ~75% of Z, not 100%. Root cause: the bigram matrix is sparsified with top-p=0.95, so the stationary distribution (unigram) doesn't match the true LM marginal. The marginalization identity `Σ_v P(s₁|v) × P_unigram(v) = P_unigram(s₁)` only holds for the exact stationary distribution of the full (untruncated) transition matrix.

**Consequence**: All ~50k candidates are evaluated every backward step. With batch_size=1024 on GPU, this takes ~30s per step for short suffixes, scaling linearly with suffix length.

**Fix**: Recompute bigrams without top-p sparsification. This will produce a denser matrix but the unigram will correctly normalize, enabling early stopping. Alternatively, use a lower threshold (e.g., 0.7).

### 2. Output quality is mediocre
Sample outputs for suffix " is my favorite movie" (temperature=0.5):
- "to say, this is my favorite movie" (good)
- "be ago, this is my favorite movie" (ok)
- ".t.. is my favorite movie" (bad)

The posterior distribution over predecessor tokens is genuinely high-entropy — many tokens are plausible predecessors. Temperature helps but doesn't fully solve it. There may also be a bug in the probability math that needs investigation.

### 3. GPT-2 is a weak model
GPT-2 (137M params) has limited contextual understanding. A stronger model would likely produce sharper posteriors and better backward samples.

## Test Results
- 13 non-slow tests pass (bigram shape, unigram convergence, artifact I/O, log-Z, sampling utilities)
- Slow tests (requiring full artifacts) not yet validated with the new full-suffix scoring

## File Summary
| File | Purpose |
|------|---------|
| `src/backward_sampler/bigram.py` | Bigram precomputation, BigramArtifacts dataclass |
| `src/backward_sampler/sampler.py` | backward_step, backward_generate, compute_log_z |
| `src/backward_sampler/utils.py` | Model loading, sampling utility |
| `scripts/precompute_bigrams.py` | CLI for bigram precomputation |
| `scripts/backward_generate.py` | CLI for backward generation |
| `scripts/debug_paris.py` | Debug: score analysis for "capital of France" |
| `scripts/debug_scores.py` | Debug: score distribution analysis |
| `tests/test_sampler.py` | Unit tests |
