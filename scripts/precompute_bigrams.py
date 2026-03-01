"""Precompute bigram artifacts from a language model.

Usage:
    uv run scripts/precompute_bigrams.py --output-dir artifacts/gpt2
    uv run scripts/precompute_bigrams.py --model-name openai-community/gpt2 --device cuda --batch-size 512
"""

from __future__ import annotations

import argparse
import time

from backward_sampler.bigram import precompute_artifacts
from backward_sampler.utils import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute bigram artifacts")
    parser.add_argument(
        "--model-name",
        default="openai-community/gpt2",
        help="HuggingFace model identifier (default: openai-community/gpt2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for forward passes (default: 256)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on: cpu, cuda, cuda:0, etc. (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/gpt2",
        help="Directory to save artifacts (default: artifacts/gpt2)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling threshold for sparsification (default: 0.95)",
    )
    parser.add_argument(
        "--power-iterations",
        type=int,
        default=100,
        help="Number of power iteration steps for unigram (default: 100)",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name, args.device)
    vocab_size = model.config.vocab_size
    print(f"Vocab size: {vocab_size}")

    print(f"Starting bigram precomputation (batch_size={args.batch_size})...")
    start = time.time()
    artifacts = precompute_artifacts(
        model=model,
        vocab_size=vocab_size,
        batch_size=args.batch_size,
        top_p=args.top_p,
        power_iterations=args.power_iterations,
    )
    elapsed = time.time() - start
    print(f"Precomputation completed in {elapsed:.1f}s")

    nnz = artifacts.log_bigram_csr.nnz
    density = nnz / (vocab_size * vocab_size)
    print(f"Bigram matrix: {nnz:,} nonzero entries ({density:.4%} density)")

    print(f"Saving artifacts to {args.output_dir}")
    artifacts.save(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
