"""Run backward generation from a suffix.

Usage:
    uv run scripts/backward_generate.py --suffix "is the capital of France"
    uv run scripts/backward_generate.py --suffix "is the capital of France" --max-tokens 10 --verbose
"""

from __future__ import annotations

import argparse

from backward_sampler.bigram import BigramArtifacts
from backward_sampler.sampler import backward_generate
from backward_sampler.utils import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Backward generation from a suffix")
    parser.add_argument(
        "--suffix",
        required=True,
        help="The suffix text to generate backward from",
    )
    parser.add_argument(
        "--model-name",
        default="openai-community/gpt2",
        help="HuggingFace model identifier (default: openai-community/gpt2)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="artifacts/gpt2",
        help="Directory containing precomputed bigram artifacts (default: artifacts/gpt2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=20,
        help="Maximum number of tokens to prepend (default: 20)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Nucleus stopping threshold (default: 0.9)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of independent samples to generate (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each generated token",
    )
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name, args.device)

    print(f"Loading artifacts from: {args.artifacts_dir}")
    artifacts = BigramArtifacts.load(args.artifacts_dir)

    print(f"Suffix: {args.suffix!r}")
    print(f"Generating {args.num_samples} samples (max {args.max_tokens} tokens each)...")
    print()

    for i in range(args.num_samples):
        result = backward_generate(
            model=model,
            tokenizer=tokenizer,
            suffix=args.suffix,
            artifacts=artifacts,
            max_new_tokens=args.max_tokens,
            threshold=args.threshold,
            verbose=args.verbose,
        )
        print(f"[{i + 1}] {result}")


if __name__ == "__main__":
    main()
