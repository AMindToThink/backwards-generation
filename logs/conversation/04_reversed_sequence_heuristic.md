# Conversation: 04_reversed_sequence_heuristic

## User

[Request interrupted by user for tool use]

## User

Implement the following plan:

# Plan: Reversed-Sequence Heuristic for Candidate Ordering

## Context
The cloze-style forward-prompt heuristic works but is slow (~1 min/step) because its ordering doesn't concentrate Bayesian mass in the first batches. New idea: feed the model a long passage with tokens reversed, then append the reversed suffix. The model's next-token prediction on reversed text predicts the predecessor in the original order. This uses in-context learning of one long reversed example rather than a few-shot cloze prompt.

## Changes

### 1. Add `_get_reversed_prompt_order()` to `src/backward_sampler/sampler.py`

New function alongside `_get_forward_prompt_order()` (~line 143):

- Store a hardcoded ~200-token passage as `_REVERSED_PASSAGE` (module-level string constant, e.g., opening of a well-known Wikipedia article)
- Tokenize it, reverse the token list → `reversed_passage_ids`
- Append the reversed suffix: `reversed_passage_ids + suffix_ids[::-1]`
- If combined length > 1024 (GPT-2 context), truncate passage from the **left** (keep the suffix intact — it's what matters)
- Forward pass, take logits at last position, sort all vocab tokens descending
- Return `np.ndarray` of all token IDs (same interface as other heuristics)

### 2. Update `backward_step()` in `src/backward_sampler/sampler.py`

- Extend `heuristic` Literal: `"bigram" | "forward-prompt" | "reversed"`
- Add `elif heuristic == "reversed":` dispatch calling `_get_reversed_prompt_order(model, tokenizer, suffix_ids)`
- Same assertion (must return all vocab tokens)

### 3. Update `backward_generate()` in `src/backward_sampler/sampler.py`

- Update `heuristic` Literal type to include `"reversed"`

### 4. Update CLI in `scripts/backward_generate.py`

- Add `"reversed"` to `--heuristic` choices

### 5. Add tests to `tests/test_sampler.py`

- `TestReversedPromptOrder`: 4 unit tests (returns full vocab, valid IDs, no duplicates, numpy array)

### Files modified
- `src/backward_sampler/sampler.py`
- `scripts/backward_generate.py`
- `tests/test_sampler.py`

## Verification
1. `uv run pytest tests/ -v -m "not slow"` — all tests pass
2. GPU run (confirm with Matthew first):
   ```
   uv run python scripts/backward_generate.py \
     --model-name openai-community/gpt2-xl \
     --suffix " is the capital of France" \
     --device cuda:0 --batch-size 512 --threshold 0.9 \
     --temperature 0.5 --num-samples 3 --max-tokens 5 \
     --heuristic reversed
   ```


If you need specific details from before exiting plan mode (like exact code snippets, error messages, or content you generated), read the full transcript at: /home/cs29824/.claude/projects/-home-cs29824-matthew-backwards-generation/85b5e8d0-2c88-4081-8831-e65d8ec36775.jsonl

## Assistant

I'll start by reading the files I need to modify.

## Assistant

**Tool: Read**
```
file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py
```

## Assistant

**Tool: Read**
```
file_path: /home/cs29824/matthew/backwards-generation/scripts/backward_generate.py
```

## Assistant

**Tool: Read**
```
file_path: /home/cs29824/matthew/backwards-generation/tests/test_sampler.py
```

## Assistant

Now I have full context. Let me implement all the changes.

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py, old_string: _CLOZE_PROMPT = '''\, new_string: _REVERSED_PASSAGE = (
    "The history of the United States began with the arrival of Native Americans"
    " in North America around 15,000 BC. Numerous indigenous cultures formed, and"
    " many sa...
```

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py, old_string: def _score_candidate_batch(, new_string: def _get_reversed_prompt_order(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    suffix_ids: list[int],
) -> np.ndarray:
    """Get candidate predecessor tokens by feeding rever...
```

## Assistant

Now update `backward_step` to support the new heuristic:

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py, old_string:     heuristic: Literal["bigram", "forward-prompt"] = "bigram",
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> int:
    """Sample a single token to prepend to the suffix.

    Implements th..., new_string:     heuristic: Literal["bigram", "forward-prompt", "reversed"] = "bigram",
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> int:
    """Sample a single token to prepend to the suffix.

    I...
```

## Assistant

Now update the dispatch logic:

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py, old_string:     # Step 2: Get candidate order from heuristic
    if heuristic == "forward-prompt":
        candidates = _get_forward_prompt_order(model, tokenizer, suffix_ids)
        # DO NOT REMOVE: The heurist..., new_string:     # Step 2: Get candidate order from heuristic
    if heuristic == "forward-prompt":
        candidates = _get_forward_prompt_order(model, tokenizer, suffix_ids)
        # DO NOT REMOVE: The heurist...
```

## Assistant

Now update `backward_generate`:

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py, old_string:     heuristic: Literal["bigram", "forward-prompt"] = "bigram",
) -> str:, new_string:     heuristic: Literal["bigram", "forward-prompt", "reversed"] = "bigram",
) -> str:
```

## Assistant

Now update the CLI:

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/scripts/backward_generate.py, old_string:         choices=["bigram", "forward-prompt"],, new_string:         choices=["bigram", "forward-prompt", "reversed"],
```

## Assistant

Now add the tests:

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/tests/test_sampler.py, old_string: from backward_sampler.sampler import (
    _get_forward_prompt_order,
    _score_candidate_batch,
    backward_generate,
    backward_step,
    compute_log_z,
), new_string: from backward_sampler.sampler import (
    _get_forward_prompt_order,
    _get_reversed_prompt_order,
    _score_candidate_batch,
    backward_generate,
    backward_step,
    compute_log_z,
)
```

## Assistant

Now add the test class at the end of the file:

## Assistant

**Tool: Read**
```
file_path: /home/cs29824/matthew/backwards-generation/tests/test_sampler.py, offset: 360
```

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/tests/test_sampler.py, old_string:     def test_requires_tokenizer(self, model, tokenizer, small_artifacts):
        """forward-prompt heuristic should raise ValueError without tokenizer."""
        suffix_ids = tokenizer.encode(" test..., new_string:     def test_requires_tokenizer(self, model, tokenizer, small_artifacts):
        """forward-prompt heuristic should raise ValueError without tokenizer."""
        suffix_ids = tokenizer.encode(" test...
```

## Assistant

Now let's run the tests:

## Assistant

**Tool: Bash**
```
command: uv run pytest tests/ -v -m "not slow" 2>&1, description: Run non-slow tests, timeout: 300000
```

## Assistant

**Tool: Bash**
```
command: uv run pytest tests/ -v -m "not slow" 2>&1, description: Run non-slow tests, timeout: 300000, dangerouslyDisableSandbox: True
```

## Assistant

All 24 tests pass (5 slow tests deselected). Here's a summary of what was implemented:

**`src/backward_sampler/sampler.py`:**
- Added `_REVERSED_PASSAGE` — a ~200-token Wikipedia-style passage used as in-context learning material
- Added `_get_reversed_prompt_order()` — tokenizes the passage, reverses it, appends reversed suffix, runs a forward pass, and sorts all vocab tokens by descending logits at the last position
- Left-truncates the passage (keeping suffix intact) if combined length exceeds 1024 tokens
- Updated `backward_step()` and `backward_generate()` type hints to accept `"reversed"` heuristic
- Added dispatch logic with the same full-vocab assertion as forward-prompt

**`scripts/backward_generate.py`:**
- Added `"reversed"` to `--heuristic` choices

**`tests/test_sampler.py`:**
- Added `TestReversedPromptOrder` with 4 tests: full vocab coverage, valid IDs, no duplicates, numpy return type

Want me to do a GPU test run, or commit this?

## User

run it

## Assistant

This needs GPU. You're good to run on `cuda:0` right now?

The command will be:
```
uv run python scripts/backward_generate.py \
  --model-name openai-community/gpt2-xl \
  --suffix " is the capital of France" \
  --device cuda:0 --batch-size 512 --threshold 0.9 \
  --temperature 0.5 --num-samples 3 --max-tokens 5 \
  --heuristic reversed
```

## User

yes, I'm good

## Assistant

**Tool: Bash**
```
command: uv run python scripts/backward_generate.py --model-name openai-community/gpt2-xl --suffix " is the capital of France" --device cuda:0 --batch-size 512 --threshold 0.9 --temperature 0.5 --num-samples 3..., description: Run backward generation with reversed heuristic on GPU, timeout: 600000, dangerouslyDisableSandbox: True
```

## Assistant

Good question. Let me check the ordering while the GPU run completes.

## Assistant

**Tool: Read**
```
file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py, offset: 290, limit: 50
```

## Assistant

**Tool: Read**
```
file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py, offset: 339, limit: 20
```

## Assistant

Temperature is applied **after** the nucleus cutoff (line 353), not before. The stopping criterion on line 349 uses the raw (untempered) log scores to decide when enough probability mass has been accumulated, then temperature is only applied for the final sampling step.

This is actually the correct design — the nucleus stopping criterion is answering "have we found enough of the true posterior mass?", which is a question about the model's actual probabilities. Temperature is a sampling knob that reshapes the distribution you sample from, but shouldn't affect *which* candidates you include.

If you applied temperature *before* the cutoff, a low temperature (like 0.5) would sharpen the scores and cause `logsumexp` to be dominated by the top candidates, reaching the threshold sooner — but you'd be cutting off based on a distorted distribution rather than the true posterior mass. You'd potentially miss candidates that have meaningful true probability.

So the current ordering is intentional and correct: gather candidates until you have 90% of the true mass, then temperature-shape and sample.

Let me check on the GPU run.

## Assistant

**Tool: TaskOutput**
```
task_id: b0qjrqxa2, block: True, timeout: 600000
```

## Assistant

Results look good — all 3 samples correctly found "Paris":

```
[1]  up in Paris. Paris is the capital of France
[2]  2017 in Paris. Paris is the capital of France
[3]  in the city. Paris is the capital of France
```

And to circle back on your temperature question: temperature is applied **after** the nucleus cutoff (line 353 of `sampler.py`). The stopping criterion at line 349 uses raw log-scores to determine when 90% of the true posterior mass is captured, then temperature only reshapes the distribution for the final sample. This is the correct ordering — you want the cutoff to reflect the model's actual probabilities, not a distorted version. If you applied temperature before, low temps would let you stop earlier but you'd be cutting based on artificially concentrated mass rather than the real distribution.

## User

How did the speed compare?

## Assistant

Good question — I didn't capture timing info. Let me run it again with `time` to compare reversed vs forward-prompt.

## Assistant

**Tool: Bash**
```
command: time uv run python scripts/backward_generate.py --model-name openai-community/gpt2-xl --suffix " is the capital of France" --device cuda:0 --batch-size 512 --threshold 0.9 --temperature 0.5 --num-samp..., description: Time the reversed heuristic, timeout: 600000, dangerouslyDisableSandbox: True
```

## Assistant

**Tool: Bash**
```
command: time uv run python scripts/backward_generate.py --model-name openai-community/gpt2-xl --suffix " is the capital of France" --device cuda:0 --batch-size 512 --threshold 0.9 --temperature 0.5 --num-samp..., description: Time the forward-prompt heuristic, timeout: 600000, dangerouslyDisableSandbox: True
```

## User

[Request interrupted by user for tool use]

## User

Nah, I'm too impatient for that. Please add tqdm to the code for loading bars. After that, update the documentation, commit, push.

## Assistant

Let me check what docs exist and then make the changes.

## Assistant

**Tool: Glob**
```
pattern: **/*.md
```

## Assistant

**Tool: Bash**
```
command: uv run python -c "import tqdm" 2>&1, description: Check if tqdm is installed
```

## Assistant

**Tool: Bash**
```
command: uv run python -c "import tqdm" 2>&1, description: Check if tqdm is installed, dangerouslyDisableSandbox: True
```

## Assistant

**Tool: Read**
```
file_path: /home/cs29824/matthew/backwards-generation/README.md
```

## Assistant

tqdm is already installed. Now let me read the current `backward_step` to add a progress bar for the batch loop, and `backward_generate` for the token loop.

## Assistant

**Tool: Read**
```
file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py
```

## Assistant

Now I'll add tqdm to both loops, update the README, then commit and push.

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py, old_string: import numpy as np
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase, new_string: import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase
```

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py, old_string:     # Step 3: Score candidates in batches, stopping when nucleus threshold met
    all_log_scores: list[float] = []
    all_token_ids: list[int] = []
    accumulated_log_mass = float("-inf")  # log(0)..., new_string:     # Step 3: Score candidates in batches, stopping when nucleus threshold met
    all_log_scores: list[float] = []
    all_token_ids: list[int] = []
    accumulated_log_mass = float("-inf")  # log(0)...
```

## Assistant

Now add a progress bar to the outer token generation loop in `backward_generate`:

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/src/backward_sampler/sampler.py, old_string:     suffix_ids = tokenizer.encode(suffix)
    eos_token_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        token = backward_step(
            model,
            suffix_ids,
  ..., new_string:     suffix_ids = tokenizer.encode(suffix)
    eos_token_id = tokenizer.eos_token_id

    pbar = tqdm(range(max_new_tokens), desc="Backward steps", leave=False)
    for step in pbar:
        token = ba...
```

## Assistant

Now update the README to document the `reversed` heuristic:

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/README.md, old_string: # Generate with forward-prompt heuristic (no bigram ordering)
uv run python scripts/backward_generate.py \
  --suffix " is the capital of France" \
  --model-name openai-community/gpt2-xl \
  --device..., new_string: # Generate with forward-prompt heuristic (no bigram ordering)
uv run python scripts/backward_generate.py \
  --suffix " is the capital of France" \
  --model-name openai-community/gpt2-xl \
  --device...
```

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/README.md, old_string: | Heuristic | Description | Speed |
|-----------|-------------|-------|
| `bigram` (default) | Orders candidates by reverse bigram probability `P(v\|s₁)` from the precomputed sparse matrix. Only evalu..., new_string: | Heuristic | Description | Speed |
|-----------|-------------|-------|
| `bigram` (default) | Orders candidates by reverse bigram probability `P(v\|s₁)` from the precomputed sparse matrix. Only evalu...
```

## Assistant

**Tool: Edit**
```
replace_all: False, file_path: /home/cs29824/matthew/backwards-generation/README.md, old_string: uv run pytest tests/ -v -m "not slow"  # fast tests (20 tests), new_string: uv run pytest tests/ -v -m "not slow"  # fast tests (24 tests)
```

## Assistant

Now run the tests to make sure tqdm doesn't break anything:

## Assistant

**Tool: Bash**
```
command: uv run pytest tests/ -v -m "not slow" 2>&1, description: Run tests after tqdm changes, timeout: 300000, dangerouslyDisableSandbox: True
```

## Assistant

All 24 tests pass. Now commit and push.

## Assistant

**Tool: Bash**
```
command: git status && git diff --stat, description: Check git status and diff summary
```

## Assistant

**Tool: Bash**
```
command: git log --oneline -5, description: Check recent commits
```

## Assistant

**Tool: Bash**
```
command: git status && git diff --stat, description: Check git status, dangerouslyDisableSandbox: True
```

## Assistant

**Tool: Bash**
```
command: git log --oneline -5, description: Recent commits, dangerouslyDisableSandbox: True
```

## Assistant

**Tool: Bash**
```
command: git add README.md scripts/backward_generate.py src/backward_sampler/sampler.py tests/test_sampler.py && git commit -m "$(cat <<'EOF'
Add reversed-sequence heuristic and tqdm progress bars

New candida..., description: Commit changes, dangerouslyDisableSandbox: True
```

## Assistant

**Tool: Bash**
```
command: git push, description: Push to remote, dangerouslyDisableSandbox: True
```

## Assistant

Done. Committed and pushed. Summary of changes:

- **`sampler.py`**: Added `_get_reversed_prompt_order()` heuristic + tqdm progress bars on both the candidate scoring loop (shows coverage %) and the outer backward generation loop (shows latest token)
- **`backward_generate.py`**: Added `"reversed"` to `--heuristic` choices
- **`test_sampler.py`**: Added `TestReversedPromptOrder` with 4 tests
- **`README.md`**: Documented the new heuristic with example command and table entry

## User

<local-command-caveat>Caveat: The messages below were generated by the user while running local commands. DO NOT respond to these messages or otherwise consider them in your response unless the user explicitly asks you to.</local-command-caveat>

## User

<command-name>/exit</command-name>
            <command-message>exit</command-message>
            <command-args></command-args>

## User

<local-command-stdout>Bye!</local-command-stdout>
