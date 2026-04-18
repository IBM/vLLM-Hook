# Token Highlighter Plan (Current Implementation)

## Goal
Identify prompt tokens that are primarily responsible for increasing model's likelihood of producing an affirmative target phrase, then optionally soften those tokens during generation to reduce their influence.

## Token Highlighter Algorithm
Given prompt tokens `X` and target affirmation tokens `Y`:
1. Compute affirmation sequence loss under teacher forcing:
   - `L_aff(X) = -sum_i log P(y_i | X, y_<i)`
2. Compute per-token influence score on prompt embeddings:
   - `score_j = || dL_aff / d emb(x_j) ||_2`
3. Select driver tokens from `score_j` (two methods):
   - `mean_std`: `score > mean + k*std`
   - `top_percentage`: top `argtop-(n*alpha){scores}` for `alpha \in [0, 1]` (from original paper)
4. Apply soft-removal in prefill by scaling selected prompt embeddings with `beta` and continuing generation.

View the original paper [**Token Highlighter: Inspecting and Mitigating Jailbreak Prompts for Large Language Models**](https://arxiv.org/pdf/2412.18171) for more information.

## Worker / Analyzer Communication

### Worker (`HighlighterWorker`)
- Trigger: prefill requests in `execute_model` when hook flag is active.
- Note on autograd scorer: we load a separate Hugging Face scorer model from the same local snapshot as the serving model, because vLLM's execution path is inference-optimized and does not always expose a reliable autograd graph for custom backward passes. By loading from the same snapshot, we preserve model architecture, weights, and tokenizer during gradient computation.
- Responsibilities:
  - compute `token_scores` per prompt
  - choose applied driver indices (`soft_indices`)
  - queue softened prompt embeddings for embedding-hook injection
  - save per-run trace (`highlighter.pt`)
- Per-sequence trace fields:
  - `token_ids`
  - `tokens` (decoded when tokenizer is available)
  - `token_scores`
  - `soft_indices` (applied during generation)
  - `soft_beta`
  - applied selection metadata (`applied_mode`, `applied_threshold_k`, `applied_alpha`)

### Analyzer (`HighlighterAnalyzer`)
- Reads latest `highlighter.pt` for the run.
- Returns:
  - `drivers`: worker-applied `soft_indices`
  - `analysis_drivers`: optional analyzer-side re-selection from `token_scores` using `analyzer_spec`
  - `top_tokens`: top-k by score
- No gradient recomputation in analyzer.

## Demo Structure (`examples/demo_token_highlighter.py`)
1. Resolve local model snapshot and set environment configuration.
2. Precompute and export `VLLM_HIGHLIGHTER_TARGET_TOKEN_IDS`.
3. Call `llm.generate(prompt)` once per prompt.
4. Call `llm.analyze(analyzer_spec=...)` for reporting (can be repeated with different specs).
5. Print applied driver tokens, optional analyzer-side driver tokens (if different from the worker's selected tokens, chosen via `_select_analysis_drivers()`), and top tokens by score.

## Design Intent
- Keep mitigation decisions in worker (generation-time behavior).
- Keep analysis flexibility in analyzer (post-hoc views from stored scores).
- Allow users to run one generation and inspect multiple analysis specs without recomputing gradients.

## Important Notes
- The registry file `vllm_hook_plugins/vllm_hook_plugins/__init__.py` already registers the Token Highlighter as a worker and analyzer.