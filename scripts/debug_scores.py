"""Debug: examine score distribution for a suffix."""
from backward_sampler.utils import load_model
from backward_sampler.bigram import BigramArtifacts
from backward_sampler.sampler import compute_log_z, _get_candidate_order, _score_candidate_batch
import torch
import numpy as np

model, tokenizer = load_model("openai-community/gpt2", "cuda:0")
artifacts = BigramArtifacts.load("artifacts/gpt2")

suffix = " is my favorite movie"
suffix_ids = tokenizer.encode(suffix)
print(f"Suffix: {suffix!r}")
print(f"Suffix tokens: {suffix_ids} = {[tokenizer.decode([t]) for t in suffix_ids]}")

log_z = compute_log_z(model, suffix_ids, artifacts.log_unigram)
print(f"log_z = {log_z:.4f}")

s1 = suffix_ids[0]
candidates = _get_candidate_order(artifacts, s1)

# Score ALL candidates
all_scores: list[float] = []
all_ids: list[int] = []
batch_size = 1024

for batch_start in range(0, len(candidates), batch_size):
    batch_end = min(batch_start + batch_size, len(candidates))
    batch_tokens = candidates[batch_start:batch_end]
    scores, ids = _score_candidate_batch(model, batch_tokens, suffix_ids, artifacts.log_unigram)
    all_scores.extend(scores)
    all_ids.extend(ids)

# Sort by score
scored = sorted(zip(all_scores, all_ids), reverse=True)

print("\nTop 30 candidates:")
for rank, (score, v) in enumerate(scored[:30]):
    text = tokenizer.decode([v])
    # Decompose: what's the suffix prob vs unigram?
    log_p_v = float(artifacts.log_unigram[v])
    log_p_suffix_v = score - log_p_v
    print(f"  #{rank}: {text!r} (id={v}) total={score:.2f}  P(suffix|v)={log_p_suffix_v:.2f}  P(v)={log_p_v:.2f}")

# Normalize and show sampling probabilities for top 10
top_scores = torch.tensor([s for s, _ in scored[:30]])
top_probs = torch.softmax(top_scores, dim=0)
print("\nSampling probabilities (top 30 after normalization):")
for rank, ((score, v), prob) in enumerate(zip(scored[:30], top_probs)):
    text = tokenizer.decode([v])
    print(f"  #{rank}: {text!r} prob={prob:.4f}")
