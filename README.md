# Backward Sampler

Bayesian backward token sampling from autoregressive language models. Given a suffix like `" is the capital of France"`, this system samples plausible predecessor tokens to generate text that ends with the given suffix.

## How it works

Standard autoregressive LMs generate left-to-right. This project inverts the process using Bayes' rule:

```
P(v | suffix) ∝ P(suffix | v) × P(v)
```

where:
- **P(suffix | v)** is computed via a forward pass on `[v, s₁, ..., s_T]`
- **P(v)** is the stationary distribution of the model's bigram Markov chain

Each backward step samples a token to prepend, then the new token becomes part of the suffix for the next step.

### Pipeline

1. **Bigram precomputation** (`scripts/precompute_bigrams.py`): Pass every token through the LM to build a sparse bigram transition matrix. Compute the stationary distribution via power iteration.
2. **Backward generation** (`scripts/backward_generate.py`): For each step, score candidate predecessors using the full LM and sample from the Bayesian posterior.

## Setup

```bash
# Install dependencies
uv sync --extra dev

# Precompute bigram artifacts (requires GPU, ~7 min)
uv run python scripts/precompute_bigrams.py \
  --model-name openai-community/gpt2 \
  --output-dir artifacts/gpt2 \
  --device cuda:0

# Generate backward from a suffix (bigram heuristic, default)
uv run python scripts/backward_generate.py \
  --suffix " is the capital of France" \
  --model-name openai-community/gpt2-xl \
  --device cuda:0 \
  --batch-size 512 \
  --temperature 0.5 \
  --num-samples 5 \
  --max-tokens 5 \
  --stream

# Generate with forward-prompt heuristic (no bigram ordering)
uv run python scripts/backward_generate.py \
  --suffix " is the capital of France" \
  --model-name openai-community/gpt2-xl \
  --device cuda:0 \
  --batch-size 512 \
  --threshold 0.9 \
  --temperature 0.5 \
  --num-samples 3 \
  --max-tokens 5 \
  --heuristic forward-prompt
```

## Candidate ordering heuristics

The `--heuristic` flag controls how candidates are ordered for evaluation. The ordering only affects **search order** — actual sampling probabilities always use the Bayesian scoring (`P(suffix|v) × P(v)`), and the nucleus stopping criterion decides when to stop.

| Heuristic | Description | Speed |
|-----------|-------------|-------|
| `bigram` (default) | Orders candidates by reverse bigram probability `P(v\|s₁)` from the precomputed sparse matrix. Only evaluates tokens with nonzero bigram probability. | Fast when bigrams are unsparsified |
| `forward-prompt` | Uses a few-shot cloze prompt to get the LM's own prediction for what precedes the suffix. Evaluates all ~50k tokens, sorted by heuristic probability. | ~1 extra forward pass per step |

## Example output

**Suffix:** `" is the capital of France"` (GPT-2 XL, temperature 0.5, forward-prompt heuristic)
```
[1]  Paris in Paris. Paris is the capital of France
[2]  live in Paris. Paris is the capital of France
[3]  Paris.\n\nParis is the capital of France
```

**Suffix:** `" is the name of my favorite movie"` (GPT-2 XL, temperature 0.5)
```
[1]  you. Yes, that is the name of my favorite movie
[2]  you to know that that is the name of my favorite movie
[3]  man. Yes, that is the name of my favorite movie
[4] .Stitchin' is the name of my favorite movie
[5]  this.\nThis is the name of my favorite movie
```

## Model compatibility

Any HuggingFace causal LM works. The bigram artifacts are tokenizer-specific, so models sharing the same tokenizer (e.g., GPT-2 / GPT-2 XL) can share artifacts.

| Model | Params | Vocab | Notes |
|-------|--------|-------|-------|
| `openai-community/gpt2` | 137M | 50,257 | Fast, baseline |
| `openai-community/gpt2-xl` | 1.6B | 50,257 | Sharper posteriors, same artifacts |
| `HuggingFaceTB/SmolLM-1.7B` | 1.7B | 49,152 | Needs separate bigram precomputation |

## Tests

```bash
uv run pytest tests/ -v -m "not slow"  # fast tests (20 tests)
uv run pytest tests/ -v -m slow        # requires precomputed artifacts
```
