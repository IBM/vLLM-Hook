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
- `export_metal_bundle.py`
  Runs the local Metal example runner against the same debug JSON input and
  copies the resulting `qkv.pt` into a stable bundle path under `output/`.
- `compare_bundle_to_metal.py`
  Runs the local comparison against a Metal `qkv.pt`. If `--metal` is omitted,
  it uses `output/metal_bundle_<analyzer>/qkv.pt`.
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

For `attntracker`, either use the checked-in sample file:

```text
sandbox/colab_sandbox/attntracker_input.json
```

or create `/content/attntracker_input.json` with:

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

For `corer`, either use the checked-in sample file:

```text
sandbox/colab_sandbox/corer_input.json
```

or create `/content/corer_input.json` with:

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

You can also use the same files locally with the debug-only example runner path:

```bash
cd /Users/timothyburley/opensource/vLLM-Hook
VLLM_HOOK_BACKEND=metal python examples/demo_corer.py --debug-input-json sandbox/colab_sandbox/corer_input.json
```

```bash
cd /Users/timothyburley/opensource/vLLM-Hook
VLLM_HOOK_BACKEND=metal python examples/demo_attntracker.py --debug-input-json sandbox/colab_sandbox/attntracker_input.json
```

### 2b. Export a repeatable local Metal bundle

For `corer`:

```bash
cd /Users/timothyburley/opensource/vLLM-Hook
/Users/timothyburley/opensource/.venv/bin/python sandbox/colab_sandbox/export_metal_bundle.py \
  --analyzer corer \
  --input-json sandbox/colab_sandbox/corer_input.json \
  --output-dir sandbox/colab_sandbox/output/metal_bundle_corer
```

For `attntracker`:

```bash
cd /Users/timothyburley/opensource/vLLM-Hook
/Users/timothyburley/opensource/.venv/bin/python sandbox/colab_sandbox/export_metal_bundle.py \
  --analyzer attntracker \
  --input-json sandbox/colab_sandbox/attntracker_input.json \
  --output-dir sandbox/colab_sandbox/output/metal_bundle_attntracker
```

### 3. Export the non-Metal bundle

For `attntracker`:

```bash
python sandbox/colab_sandbox/export_nonmetal_bundle.py \
  --analyzer attntracker \
  --model ibm-granite/granite-4.0-micro \
  --input-json /content/attntracker_input.json \
  --gpu-memory-utilization 0.2 \
  --max-model-len 1024 \
  --output-dir /content/nonmetal_bundle_attn
```

For `corer`:

```bash
python sandbox/colab_sandbox/export_nonmetal_bundle.py \
  --analyzer corer \
  --model ibm-granite/granite-4.0-micro \
  --input-json /content/corer_input.json \
  --gpu-memory-utilization 0.2 \
  --max-model-len 1024 \
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

### If Colab still refuses to start the engine

The key line is:

```text
Free memory on device cuda:0 (...) is less than desired GPU memory utilization
```

That means the GPU already has too little free memory for the fraction you
requested. On Colab this usually happens because an earlier vLLM engine is
still alive in the notebook session.

Use this recovery sequence:

```bash
pkill -f vllm || true
pkill -f EngineCore || true
python - <<'PY'
import gc
import torch
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print(torch.cuda.mem_get_info())
PY
```

Then rerun the exporter with a lower memory target if needed:

```bash
python sandbox/colab_sandbox/export_nonmetal_bundle.py \
  --analyzer attntracker \
  --model ibm-granite/granite-4.0-micro \
  --input-json /content/attntracker_input.json \
  --gpu-memory-utilization 0.15 \
  --max-model-len 1024 \
  --output-dir /content/nonmetal_bundle_attn
```

If that still fails, restart the Colab runtime before retrying.

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
  --bundle-dir /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_corer \
  --output-dir /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output
```

That command will automatically use:

```text
/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/metal_bundle_corer/qkv.pt
```

for a CoRe bundle, or the corresponding `metal_bundle_attntracker/qkv.pt` path
for AttentionTracker.

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

## Repeatable Test Commands

This is the shortest repeatable sequence using the checked-in JSON payloads.

### CoRe: Colab non-Metal export

On Colab:

```bash
git clone https://github.com/tburleyinfo/vLLM-Hook/
cd vLLM-Hook
git switch sandbox
pip install -e vllm_hook_plugins/
```

```bash
python sandbox/colab_sandbox/export_nonmetal_bundle.py \
  --analyzer corer \
  --model ibm-granite/granite-4.0-micro \
  --input-json sandbox/colab_sandbox/corer_input.json \
  --gpu-memory-utilization 0.2 \
  --max-model-len 1024 \
  --output-dir /content/nonmetal_bundle_corer
```

```bash
cd /content && zip -r nonmetal_bundle_corer.zip nonmetal_bundle_corer
```

Download:

```text
/content/nonmetal_bundle_corer.zip
```

### CoRe: local Metal export

On the local machine:

```bash
cd /Users/timothyburley/opensource/vLLM-Hook
/Users/timothyburley/opensource/.venv/bin/python sandbox/colab_sandbox/export_metal_bundle.py \
  --analyzer corer \
  --input-json sandbox/colab_sandbox/corer_input.json \
  --output-dir sandbox/colab_sandbox/output/metal_bundle_corer
```

Unzip the downloaded non-Metal bundle into:

```text
/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_corer
```

### CoRe: local comparison

```bash
/Users/timothyburley/opensource/.venv/bin/python \
  /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/compare_bundle_to_metal.py \
  --bundle-dir /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_corer \
  --output-dir /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output
```

Expected inputs used by that command:

```text
Metal:    sandbox/colab_sandbox/output/metal_bundle_corer/qkv.pt
NonMetal: sandbox/colab_sandbox/output/nonmetal_bundle_corer/qk.pt
```

### AttentionTracker: Colab non-Metal export

On Colab:

```bash
git clone https://github.com/tburleyinfo/vLLM-Hook/
cd vLLM-Hook
git switch sandbox
pip install -e vllm_hook_plugins/
```

```bash
python sandbox/colab_sandbox/export_nonmetal_bundle.py \
  --analyzer attntracker \
  --model ibm-granite/granite-4.0-micro \
  --input-json sandbox/colab_sandbox/attntracker_input.json \
  --gpu-memory-utilization 0.2 \
  --max-model-len 1024 \
  --output-dir /content/nonmetal_bundle_attn
```

```bash
cd /content && zip -r nonmetal_bundle_attn.zip nonmetal_bundle_attn
```

Download:

```text
/content/nonmetal_bundle_attn.zip
```

### AttentionTracker: local Metal export

On the local machine:

```bash
cd /Users/timothyburley/opensource/vLLM-Hook
/Users/timothyburley/opensource/.venv/bin/python sandbox/colab_sandbox/export_metal_bundle.py \
  --analyzer attntracker \
  --input-json sandbox/colab_sandbox/attntracker_input.json \
  --output-dir sandbox/colab_sandbox/output/metal_bundle_attntracker
```

Unzip the downloaded non-Metal bundle into:

```text
/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_attn
```

### AttentionTracker: local comparison

```bash
/Users/timothyburley/opensource/.venv/bin/python \
  /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/compare_bundle_to_metal.py \
  --bundle-dir /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_attn \
  --output-dir /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output
```

Expected inputs used by that command:

```text
Metal:    sandbox/colab_sandbox/output/metal_bundle_attntracker/qkv.pt
NonMetal: sandbox/colab_sandbox/output/nonmetal_bundle_attn/qk.pt
```

## Notes

- The comparison is only meaningful if Metal and non-Metal use the same
  model, tokenizer, prompt/example, and layer/head shortlist.
- The tensor artifacts alone are not enough to reproduce analyzer behavior.
  That is why the Colab bundle includes analyzer specs and analyzer outputs.
