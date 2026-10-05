# Long-decode capture demos (Granite-8B)

Variants of `examples/demo_attntracker.py` and `examples/demo_hiddenstate.py` that make the
capture cost large enough to measure:

- **Capture in both phases.** `SamplingParams(extra_args={"hooks_on": "both"})` (the workers
  default to `"prefill"`; `extra_xargs` in the server block), so the hook fires on every decode step
  and its cost accumulates over the decode.
- **Long decode.** `max_tokens` defaults to 128, but the stock prompt reaches EOS after about 9
  tokens; use an open-ended prompt (or `ignore_eos`) for a long decode.

## Scripts and configs

`last_token` vs `all_tokens` is chosen by the config file, as in the stock demos:

| Task | Script | `MIA_CONFIG_FILE` |
|---|---|---|
| qk · last_token | `demo_attntracker_longdec.py` | *(default)* `model_configs/attention_tracker/granite-3.1-8b-instruct.json` |
| qk · all_tokens | `demo_attntracker_longdec.py` | `model_configs/attention_tracker/granite-3.1-8b-instruct_alltok.json` |
| hs · last_token | `demo_hiddenstate_longdec.py` | *(default)* `model_configs/hidden_states/granite-3.1-8b-instruct.json` |
| hs · all_tokens | `demo_hiddenstate_longdec.py` | `model_configs/hidden_states/granite-3.1-8b-instruct_alltok.json` |

## Environment variables

| Var | Default | Meaning |
|---|---|---|
| `MIA_DEMO_MODEL` | `ibm-granite/granite-3.1-8b-instruct` | HF model id |
| `MIA_CONFIG_FILE` | per-script last_token config | swap to `*_alltok.json` for all_tokens |
| `MIA_DEMO_MAX_TOKENS` | `128` | decode length |
| `MIA_DEMO_HOOKS_ON` | `both` | `prefill` \| `decode` \| `both` |

## Running

Run them like any other example (offline; each keeps its `vllm serve` version as a commented
block), from the repo root:

```bash
MIA_CONFIG_FILE=model_configs/attention_tracker/granite-3.1-8b-instruct_alltok.json \
    python examples/profiling_longdecode/demo_attntracker_longdec.py
```

`hooks_on=both` with a 128-token decode captures on every step, so artifacts grow with decode length.
