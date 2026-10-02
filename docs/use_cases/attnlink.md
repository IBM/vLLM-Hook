# AttnLink-U: Attention-Based Schema Linking

**Contributor:** [@Songjw133](https://github.com/Songjw133)

**Method:** [AttnLink: Turning Attention into Schema Links for Text-to-SQL](https://arxiv.org/abs/2608.00693)

## What it does

Given a natural-language question and a database schema, AttnLink-U ranks the
columns needed to answer the question, then selects a set by cumulative probability mass. It reads the model's attention during a
single prefill pass, without task-specific training or generating a SQL query.
The demo asks the model to copy a relevant `column@table` identifier; the ranking
comes from attention at the generation anchor, not from the sampled output.

The included configuration uses **Qwen/Qwen2.5-Coder-7B-Instruct, layer 22,
head 12** (zero-based), the model-specific pair reported in the paper. Changing
the backbone also requires a suitable layer/head configuration; `--model` is
primarily for supplying a local copy of this checkpoint.

## Quick start

Install vLLM-Hook as described in the repository README. From the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/demo_attnlink.py
```

The model is downloaded on first use. To use local weights and a named output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python examples/demo_attnlink.py \
  --model /path/to/Qwen2.5-Coder-7B-Instruct \
  --out-dir cache/attnlink_my_run
```

Each run requires a new output directory; the default includes a timestamp.
`result.json` contains every candidate's score, probability, cumulative probability,
gold/selected flags, the selected set, precision/recall/F1, AP, provenance and versions. `--gpu-memory-utilization` defaults to `0.8` and can be adjusted
for the available GPU. The demo adds no dependencies beyond the existing stack.

The sample prompt and gold columns are embedded in `examples/demo_attnlink.py`;
no separate input file is required. Defaults are `--temperature 1.0 --top-p 0.8`.
To retain more probability mass, for example:

```bash
python examples/demo_attnlink.py --temperature 1.0 --top-p 0.9
```

## How it integrates

The demo reuses the unmodified `probe_hook_qk` worker and registers
`AttnLinkAnalyzer` as `attnlink`. The model config requests the last-token Q and
full prompt K for layer 22. The worker captures the layer's QK; the analyzer
selects head 12 and its corresponding grouped-query attention K head.

The score of a candidate is the mean of its identifier-token attention values:

```text
attention = softmax(Q_anchor @ K_prompt.T * attention_scale)
score(candidate) = mean(attention[start_token:end_token])
```

Softmax is computed over the complete prompt in float32. Candidate spans are
half-open and include the complete `column@table` identifier. The prompt's final
Candidate Columns block is used, excluding the earlier in-context example.
The exact token IDs used for span alignment are supplied to inference.

The analyzer receives no gold labels. Scores retain candidate input order;
`ranking` contains indices in descending score order (ties retain input order).
After pooling, the analyzer applies the paper's candidate-set normalization:

```text
probability_i = softmax(log(score_i + 1e-8) / temperature)
selected = shortest ranked prefix with cumulative_probability >= top_p
```

At temperature 1, this normalizes the candidate scores (with epsilon smoothing).
Higher temperature flattens the distribution; higher top-p retains more columns.
There is no fixed top-k limit. The threshold-crossing item is included, at least
one item is retained, and top-p=1 retains all candidates. The normalization uses
float64 for stable post-processing; the attention computation remains float32.
These are column-selection parameters, separate from vLLM's sampling temperature.

The result additionally contains `probabilities` in input order and `selected`
as candidate indices. Gold labels affect only the demo's evaluation. The demo
prints all 55 candidates with probability, cumulative probability, gold and
selected flags, then summarizes the selected set, precision, recall, F1 and AP.

The normal in-memory call is:

```python
outputs = llm.generate([{"prompt_token_ids": ids}], sampling_params)
result = llm.analyze(analyzer_spec={
    "candidates": candidates,
    "candidate_spans": spans,
    "prompt_length": len(ids),
    "temperature": 1.0,
    "top_p": 0.8,
}, probes=outputs[0].probes)
```

For disk artifacts, generate with `save_to_disk=True, run_id="my_run"` and call
`llm.analyze(analyzer_spec=spec, run_id="my_run")`. Both paths use the same scorer
and the existing upstream QK loading utilities. To explore other settings after
one inference, reuse the scores without running the model again:

```python
from vllm_hook_plugins.analyzers.attnlink_analyzer import select_columns
selection = select_columns(result["scores"], temperature=1.0, top_p=0.9)
```

This small example handles **one prompt, one layer/head, one GPU**, with eager
execution, prefix caching disabled and chunked prefill disabled. Generation uses
`max_tokens=1` to complete the vLLM request; only prefill QK is analyzed. It does
not modify workers, serving infrastructure or model weights. Missing captures,
incomplete prompt keys and invalid spans raise errors rather than returning a
partial ranking.

## Example and validation

The included BIRD example asks for districts ranked by their number of female
account holders. It involves a join, a gender condition, grouping and counting:
**8 tables, 55 candidate columns, 5 gold columns, 1,886 rendered tokens**.
The full schema and copying prompt are preserved.

This is a selected demonstration, not a benchmark-wide accuracy claim. Temperature
1.0 and top-p=0.8 retain all five gold columns and three extra columns on this
example. The default top-p is an illustrative operating point for this sample,
not a threshold validated over the entire benchmark.

A verified run produced the following excerpt (the CLI displays all candidates):

```text
Temperature: 1 | Top-p: 0.8
Rank  Column                 Prob.      Cum.      Gold  Selected
   1  client.district_id     19.3972%   19.3972%     *      *
   2  client.client_id       17.6548%   37.0520%     *      *
   3  district.a2            12.8877%   49.9397%     *      *
   4  client.gender          10.8520%   60.7917%     *      *
   5  district.district_id    6.8162%   67.6079%     *      *
   6  account.district_id     6.7374%   74.3453%     -      *
   7  district.a5             3.4229%   77.7682%     -      *
   8  account.account_id      3.1445%   80.9127%     -      *
   9  client.birth_date       2.5045%   83.4172%     -      -
...
Selected: 8/55 | Gold covered: 5/5 | Selected mass: 80.9127%
Precision: 62.50% | Recall: 100.00% | F1: 76.92% | AP: 1.000000
```

With temperature fixed at 1.0, the same scores illustrate the trade-off:

| Top-p | Selected columns | Recall | Precision |
| --- | ---: | ---: | ---: |
| 0.75 | 7 | 100% | 71.43% |
| **0.80** | **8** | **100%** | **62.50%** |
| 0.90 | 14 | 100% | 35.71% |

Tested on an NVIDIA H100 80GB using bfloat16, `gpu_memory_utilization=0.5`,
vLLM `0.21.1.dev0+gad7125a43.d20260627.cu128`, PyTorch `2.11.0+cu128`,
Transformers `5.12.0`, and the stock QK worker from vLLM-Hook baseline
`e3d6885899264c81ac61c403c8b56efaf5a02dab` (package version `0.2.0`).

CPU checks can be run without a model download:

```bash
python tests/use_cases/test_attnlink.py -v
```

They cover GQA head selection, full-prompt normalization, span-mean pooling,
RPC/disk loading, final candidate-block alignment, AP and malformed captures,
temperature scaling, top-p boundaries, and the unchanged embedded prompt hash.

## Example provenance and license

The embedded `INPUT_SEQ` and `POSITIVE_COLS` are unchanged from zero-based index
128 of `Attnlinku/data/bird_dev.json` in the [AttnLink artifact](https://github.com/Songjw133/AttnLink).
`SOURCE` records the full dataset hash and conversion provenance. SQL and labels
are not included in the model input; gold columns are used only after scoring.

Attribution: [BIRD](https://bird-bench.github.io/) — *Can LLM Already Serve as A
Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*
(Li et al., 2023), [paper](https://arxiv.org/abs/2305.03111). Prompt conversion:
[AttnLink](https://arxiv.org/abs/2608.00693). The embedded example data retains
**CC BY-SA 4.0**, per the [BIRD license notice](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird)
and [license text](https://creativecommons.org/licenses/by-sa/4.0/legalcode).
The surrounding code retains the repository's code license.

The original sample selection kept examples with 1,500–2,500 rendered tokens,
4–8 gold columns, and at least two relevant tables. Among 466 eligible examples,
selection used descending AP, ascending distance from 1,900 tokens, then ascending
original index, with fixed Qwen2.5-Coder-7B-Instruct L22/H12. No schema columns
were removed and the layer/head was not tuned during selection.

## Citation

If you use this method in your research, please cite the AttnLink paper:

```bibtex
@article{song2026attnlink,
  title   = {AttnLink: Turning Attention into Schema Links for Text-to-SQL},
  author  = {Song, Jinwang and Liu, Tao and Zheng, Haowen and Li, Xiangheng and Li, Yifan and Zan, Hongying},
  journal = {arXiv preprint arXiv:2608.00693},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.00693}
}
```
