# Long-decode capture demos (Granite-8B)

Variants of `examples/demo_attntracker.py` and `examples/demo_hiddenstate.py` that make the
capture cost large enough to measure:

- **Capture in both phases.** `SamplingParams(extra_args={"hooks_on": "both"})` (the workers
  default to `"prefill"`; `extra_xargs` in the server block), so the hook fires on every decode step
  and its cost accumulates over the decode.
- **Long decode.** `max_tokens` defaults to 128, but the stock prompt reaches EOS after about 9
  tokens; use an open-ended prompt (or `ignore_eos`) for a long decode.

## Scripts and configs

| Task | Script | Default `MIA_CONFIG_FILE` |
|---|---|---|
| qk | `demo_attntracker_longdec.py` | `model_configs/attention_tracker/granite-3.1-8b-instruct.json` (`last_token`) |
| hs | `demo_hiddenstate_longdec.py` | `model_configs/hidden_states/granite-3.1-8b-instruct.json` (`last_token`) |

For `all_tokens`, copy the config, set `hookq_mode` (qk) or `mode` (hs) to `"all_tokens"`, and point
`MIA_CONFIG_FILE` at the copy.

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `MIA_DEMO_MODEL` | `ibm-granite/granite-3.1-8b-instruct` | HF model id |
| `MIA_CONFIG_FILE` | the script's config above | capture config |
| `MIA_DEMO_MAX_TOKENS` | `128` | decode length |
| `MIA_DEMO_HOOKS_ON` | `both` | `prefill` \| `decode` \| `both` |

## Running

Run them like the examples (offline; each keeps its `vllm serve` version as a commented block),
from the repo root:

```bash
python docs/numerical_analysis/demo_attntracker_longdec.py
```

`hooks_on=both` with a 128-token decode captures on every step, so artifacts grow with decode length.
