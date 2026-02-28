# Backward Token Sampling from Autoregressive Language Models

*Conversation Summary & Implementation Plan — February 2026*

## 1. Problem Statement

Standard autoregressive language models compute P(next token | previous tokens). Given only a suffix, we want to sample plausible tokens that could have preceded it, effectively running the model backwards. This enables applications including prompt inversion, constrained generation with fixed endings, prefix infilling, and forensic analysis of model outputs.

## 2. Core Approach: Bayesian Backward Sampling

The central insight is to use Bayes' rule to invert the forward model. Rather than computing the intractable reverse conditional directly, we decompose it into quantities the forward model can evaluate.

At each backward step, we want the probability of a candidate token v given a known suffix. By Bayes' rule, this is proportional to: the probability the model assigns to the first suffix token given v, multiplied by the prior probability of v. Both terms come from the forward model.

This transforms the problem from "invert the model" into "search over candidates and score them with the forward model," which is tractable.

## 3. Efficient Candidate Search via Bigram Proposals

Naively, scoring every token in the vocabulary (32k–128k tokens) at each step is expensive. We use a **reverse bigram table** derived from the model itself to prioritize which candidates to evaluate first.

### Bigram Table Construction

Rather than computing bigram statistics from a text corpus, we extract them directly from the language model. This is more faithful to the model's learned distribution, including any biases from post-training.

The procedure is a one-time precomputation per model:

```
# Bigram: |V| forward passes, batchable
for batch of tokens v:
    logits = model(v)
    log_bigram[v] = log_softmax(logits)

# Unigram: stationary distribution of bigram Markov chain
# Found via power iteration on the bigram transition matrix
P_0(v) = uniform
Repeat: P_{n+1}(v) = sum_w P(v|w) * P_n(w)
Until convergence
```

For a 32k vocabulary, the bigram computation takes roughly 30 minutes on a GPU with batching. The matrix is highly sparse and can be stored compressed. For 128k vocabularies, expect a few hours. Float32 precision is sufficient for the power iteration; the values remain in [0,1] and sum to 1 at each step.

### Nucleus Stopping Criterion

We adapt nucleus (top-p) sampling for the backward search. Critically, the bigram model determines only the **order of evaluation** (which candidates to try first), while the **language model** determines when to stop (whether we've captured enough probability mass). This ensures we remain faithful to the LLM's distribution even when the bigram model disagrees.

The stopping threshold uses a normalizing constant Z, which equals the LLM's probability of the current suffix. This is computed in a single forward pass at the start of each generation step.

## 4. Single-Step Algorithm

The algorithm operates on a suffix and produces one token to prepend. Repeated application grows the prefix leftward until a stopping condition is met.

**Inputs:** A suffix sequence s₁, s₂, …, s_T.

**Step 1: Compute Z.** One forward pass on (s₁, …, s_T). Accumulate log Z = Σ log P_LM(sᵢ | s₁, …, sᵢ₋₁).

**Step 2: Score candidates.** Sort vocabulary by the reverse bigram probability P_bi(v | s₁) in descending order. Iterate through candidates:

```
log_scores = {}
accumulated_mass = 0
for v in sorted vocabulary:
    log_scores[v] = log P_LM(s₁ | v) + log P_LM(v)
    accumulated_mass += exp(log_scores[v])
    if log(accumulated_mass) - log_Z > log(threshold):
        break
```

Each candidate v requires a single forward pass on the token v, from which we read off both P_LM(v) (unigram) and P_LM(s₁ | v). Candidates can be batched for efficiency.

**Step 3: Sample.** Normalize log_scores over evaluated candidates via log-softmax. Sample v. If v is BOS (beginning of sequence), stop: the model believes the sequence starts here. Otherwise, the suffix for the next step becomes (v, s₁, s₂, …, s_T).

## 5. Relationship to Prior Work

### Prompt Inversion Literature

Existing approaches to recovering prompts from LLM outputs fall into three categories, none of which use Bayesian backward sampling:

1. **Trained inversion models** (logit2prompt by Morris et al. 2023, output2prompt by Zhang et al. 2024, PILS 2025). These train a separate encoder-decoder to map outputs back to prompts.
2. **LLM-as-optimizer** (RPE/RPEGA 2024). Uses the LLM itself to iteratively refine candidate prompts via a genetic algorithm.
3. **Discrete optimization** (GCG-style, SODA). Searches over token sequences to maximize a forward-model score.

### SMC Steering

The closest existing framework is Sequential Monte Carlo (SMC) steering (Lew et al. 2023, extended by Loula et al. ICLR 2025). SMC steering frames constrained generation as posterior inference in probabilistic programs that combine LLMs with symbolic constraints. The LLaMPPL library (github.com/probcomp/hfppl) provides infrastructure for this.

Our approach is a special case of this framework where the constraint is "the suffix must equal this observed text." However, we differ in two ways: we use a cheap reverse bigram model as the proposal distribution rather than the LM itself, and we use a simple iterative sampling procedure rather than full particle-based SMC. This trades off posterior coverage for simplicity and speed.

## 6. Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| Bigram source | Extracted from LLM | Faithful to model's learned distribution; no corpus needed |
| Unigram source | Stationary distribution of bigram chain | Derived from bigrams via power iteration; avoids empty-input issue |
| Candidate ordering | Reverse bigram P_bi(v\|s₁) | Cheap; finds likely predecessors quickly |
| Stopping criterion | LLM nucleus threshold | Bigram orders, LLM decides cutoff; robust to model disagreement |
| Normalizing constant Z | Single forward pass on suffix | Recomputed each step as suffix grows |
| Arithmetic | Log-space throughout | Numerical stability |
| Stopping generation | BOS token sampled | Natural signal that model thinks sequence starts here |
| Bigram storage | Sparse matrix | Most entries near-zero; memory-efficient |

## 7. Implementation Plan

**Dependencies:** PyTorch + HuggingFace transformers (forward passes, batching), scipy.sparse (bigram matrix storage and power iteration), a HuggingFace-compatible model with logit access (e.g., LLaMA, Mistral).

### Phase 1: Bigram Precomputation

1. Batch all vocabulary tokens through the model in groups of ~256.
2. Record log-softmax output for each token. Threshold near-zero entries and store as sparse matrix.
3. Precompute sorted reverse-bigram columns: for each token w, sort all v by P(v|w) descending.
4. Run power iteration on the bigram matrix to obtain the unigram (stationary) distribution.
5. Save all artifacts to disk for reuse.

### Phase 2: Single-Step Backward Sampler

1. Implement the single-step algorithm described in Section 4.
2. Validate on simple cases: given "…is the capital of France," does it recover "Paris"?
3. Compare full unigram scoring vs. the simplified version (bigram-only, no unigram weighting) to measure impact.

### Phase 3: Multi-Step Generation Loop

1. Wrap single step in a loop. Prepend sampled token to suffix; recompute Z each iteration.
2. Implement BOS stopping condition.
3. Add optional maximum prefix length as a fallback stopping criterion.
4. Profile: measure forward passes per token, wall-clock time, and how many candidates are typically evaluated before the nucleus threshold is reached.

### Phase 4: Evaluation & Extensions

1. Evaluate on prompt inversion benchmarks (compare against output2prompt and RPE).
2. Test on constrained generation tasks (fixed endings, infilling).
3. Optionally upgrade to multi-particle SMC for better posterior coverage.
4. Explore candidate batching optimizations to reduce per-step cost.

## 8. Open Questions

- **Error compounding.** The one-step-lookahead approximation (scoring candidates only by how likely they make the next suffix token, not the entire suffix) may compound errors over long prefixes. How severe is this in practice?
- **Unigram simplification.** Does dropping the P_LM(v) unigram term and relying solely on P_LM(s₁|v) meaningfully degrade quality? This needs empirical testing.
- **Bigram matrix size.** For 128k-vocabulary models, even a sparse bigram matrix may be large. What sparsity threshold balances memory against proposal quality?
- **Multi-token scoring.** Could we periodically score candidates against the full suffix (not just the first token) to correct accumulated drift? What's the right frequency?
- **KV-cache reuse.** The Z recomputation forward pass on the growing suffix could reuse cached key-value states from previous steps. How much speedup does this provide?
