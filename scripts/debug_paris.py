"""Debug: test how many candidates are evaluated before nucleus stops,
and whether Paris gets included."""
from backward_sampler.utils import load_model
from backward_sampler.bigram import BigramArtifacts
from backward_sampler.sampler import compute_log_z, _get_candidate_order, _score_candidate_batch
import torch
import numpy as np

model, tokenizer = load_model("openai-community/gpt2", "cuda:0")
artifacts = BigramArtifacts.load("artifacts/gpt2")

suffix = " is the capital of France"
suffix_ids = tokenizer.encode(suffix)
paris_id = tokenizer.encode(" Paris")[0]

log_z = compute_log_z(model, suffix_ids, artifacts.log_unigram)
print(f"log_z = {log_z:.4f}")

s1 = suffix_ids[0]
candidates = _get_candidate_order(artifacts, s1)
print(f"Total bigram candidates: {len(candidates)}")

paris_bigram_rank = int(np.where(candidates == paris_id)[0][0])
print(f"Paris bigram rank: #{paris_bigram_rank}")

# Simulate the nucleus stopping loop with batch_size=1024
batch_size = 1024
threshold = 0.9
all_log_scores: list[float] = []
all_token_ids: list[int] = []
paris_found = False

for batch_start in range(0, len(candidates), batch_size):
    batch_end = min(batch_start + batch_size, len(candidates))
    batch_tokens = candidates[batch_start:batch_end]

    batch_scores, batch_ids = _score_candidate_batch(
        model, batch_tokens, suffix_ids, artifacts.log_unigram,
    )
    all_log_scores.extend(batch_scores)
    all_token_ids.extend(batch_ids)

    if paris_id in batch_ids and not paris_found:
        paris_found = True
        print(f"Paris found in batch {batch_start//batch_size + 1} (candidates {batch_start}-{batch_end})")

    accumulated_log_mass = torch.logsumexp(torch.tensor(all_log_scores), dim=0).item()
    ratio = accumulated_log_mass - log_z

    print(f"  After {len(all_log_scores):,} candidates: accumulated_mass - log_z = {ratio:.4f} (need > {np.log(threshold):.4f})")

    if ratio > np.log(threshold):
        print(f"\nNucleus threshold reached after {len(all_log_scores):,} candidates ({len(all_log_scores)//batch_size + 1} batches)")
        break

# Check if Paris was included
if paris_id in all_token_ids:
    idx = all_token_ids.index(paris_id)
    print(f"Paris IS in the evaluated set (index {idx})")
    # Sort and show Paris rank
    scored = sorted(zip(all_log_scores, all_token_ids), reverse=True)
    for rank, (score, v) in enumerate(scored[:10]):
        text = tokenizer.decode([v])
        marker = " <-- PARIS" if v == paris_id else ""
        print(f"  #{rank}: {text!r} score={score:.4f}{marker}")
    for rank, (score, v) in enumerate(scored):
        if v == paris_id:
            print(f"  Paris final rank: #{rank}/{len(scored)}")
            break
else:
    print(f"Paris NOT in evaluated set. Need to evaluate {paris_bigram_rank+1} candidates to reach it.")
