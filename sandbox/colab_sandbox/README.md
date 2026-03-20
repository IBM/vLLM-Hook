# Colab Sandbox

This directory is the Colab-to-local workflow for comparing a non-Metal run
captured on Colab against a Metal run captured locally.

## What You Do

1. Run one real non-Metal example on Colab.
2. Export a bundle from Colab.
3. Download that bundle to this machine.
4. Compare the downloaded non-Metal bundle against a local Metal `qkv.pt`.

## Files

- `export_nonmetal_bundle.py`
  Runs one hooked non-Metal example and exports:
  - `qk.pt`
  - `metadata.json`
  - `analyzer_outputs.json`
  - `attn_spec.json` or `corer_spec.json`
- `compare_bundle_to_metal.py`
  Runs the local comparison against a Metal `qkv.pt`.
- `output/`
  Suggested local landing zone for downloaded bundles and comparison outputs.

## Colab Steps

### 1. Clone and install

```bash
git clone https://github.com/tburleyinfo/vLLM-Hook/
cd vLLM-Hook
git switch sandbox
pip install -e vllm_hook_plugins/
```

### 2. Prepare one real example input JSON

For `attntracker`, run:

```bash
touch /content/attntracker_input.json
cat > /content/attntracker_input.json <<'EOF'
{
  "instruction": "Summarize the safety policy.",
  "data": "Policy text goes here.",
  "attn_func": "sum_normalize"
}
EOF
```

For `corer`, run:

```bash
touch /content/corer_input.json
cat > /content/corer_input.json <<'EOF'
{
  "query": "Which magazine was started first Arthur's Magazine or First for Women?",
  "documents": [
    ["Arthur's Magazine was an American literary periodical published in the 1840s."],
    ["First for Women is a woman's magazine published in the USA and was started in 1989."]
  ],
  "na_query": "N/A"
}
EOF
```

Replace those toy values with the actual example you want to compare.

### 3. Export the non-Metal bundle

For `attntracker`:

```bash
python sandbox/colab_sandbox/export_nonmetal_bundle.py \
  --analyzer attntracker \
  --model ibm-granite/granite-4.0-micro \
  --input-json /content/attntracker_input.json \
  --output-dir /content/nonmetal_bundle_attn
```

For `corer`:

```bash
python sandbox/colab_sandbox/export_nonmetal_bundle.py \
  --analyzer corer \
  --model ibm-granite/granite-4.0-micro \
  --input-json /content/corer_input.json \
  --output-dir /content/nonmetal_bundle_corer
```

### 4. Download the bundle

Download the whole directory created in `/content/`, for example:

- `/content/nonmetal_bundle_attn`
- `/content/nonmetal_bundle_corer`

Each bundle should contain:

- `qk.pt`
- `metadata.json`
- `analyzer_outputs.json`
- `attn_spec.json` or `corer_spec.json`

## Local Steps

### 1. Put the downloaded bundle under `output/`

Suggested location:

```text
/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output
```

Example:

```text
/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_corer
```

### 2. Run the local comparison

```bash
/Users/timothyburley/opensource/.venv/bin/python \
  /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/compare_bundle_to_metal.py \
  --metal /Users/timothyburley/opensource/vLLM-Hook/notebooks/~/.cache/_v1_qk_peeks/bdd91e98-ad5a-4024-a07f-7b2bf596583f/tp_rank_0/qkv.pt \
  --bundle-dir /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_corer \
  --output-dir /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output
```

That will write a local JSON summary comparing:

- raw non-Metal tensors
- raw Metal tensors
- HF reference reconstructions
- best-RoPE-offset reference variants

and, when the bundle includes analyzer spec files, it will also report
analyzer-facing score deltas.

## Recommended Order

Run two separate exports on Colab:

1. One `attntracker` example
2. One `corer` example

That lets you test the premise directly:

- do Metal and non-Metal match on AttnTracker?
- do Metal and non-Metal diverge on CoRe?

## Notes

- The comparison is only meaningful if Metal and non-Metal use the same
  model, tokenizer, prompt/example, and layer/head shortlist.
- The tensor artifacts alone are not enough to reproduce analyzer behavior.
  That is why the Colab bundle includes analyzer specs and analyzer outputs.
