# CoRe Granite 4 Micro Findings

This note summarizes the current matched Metal vs non-Metal CoRe comparison
using the repeatable `sandbox/colab_sandbox` workflow.

## Command

```bash
/Users/timothyburley/opensource/.venv/bin/python \
  /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/compare_bundle_to_metal.py \
  --bundle-dir /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_corer \
  --output-dir /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output
```

Underlying probe:

```bash
/Users/timothyburley/opensource/.venv/bin/python \
  /Users/timothyburley/opensource/vLLM-Hook/sandbox/scripts/analyzer_score_probe.py \
  --metal /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/metal_bundle_corer/qkv.pt \
  --nonmetal /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_corer/qk.pt \
  --model ibm-granite/granite-4.0-micro \
  --layer-heads 9:26;12:7,11;15:1,7,21;16:12;18:0 \
  --sample 0 \
  --max-rope-offset 16 \
  --output-dir /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output \
  --corer-spec /Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_corer/corer_spec.json
```

## Output

```text
corer_doc_sums:
  mode=nonmetal doc_sums=[0.78173828125, 2.087890625] delta_vs_nonmetal=[0.0, 0.0]
  mode=metal doc_sums=[0.80859375, 2.0625] delta_vs_nonmetal=[0.02685546875, -0.025390625]
  mode=ref_pre doc_sums=[2.609375, 2.068359375] delta_vs_nonmetal=[1.82763671875, -0.01953125]
  mode=ref_post doc_sums=[0.8056640625, 2.0703125] delta_vs_nonmetal=[0.02392578125, -0.017578125]
  mode=ref_best_metal doc_sums=[0.8056640625, 2.0703125] delta_vs_nonmetal=[0.02392578125, -0.017578125]
  mode=ref_best_nonmetal doc_sums=[0.8056640625, 2.0703125] delta_vs_nonmetal=[0.02392578125, -0.017578125]
```

## Findings

1. Metal vs non-Metal CoRe scores are close on this matched example.
   The Metal delta is about `+0.0269` on doc 1 and `-0.0254` on doc 2.

2. Pre-RoPE reconstruction is clearly wrong.
   `ref_pre` moves doc 1 by `+1.8276`, which is far outside the Metal/non-Metal
   gap.

3. Post-RoPE reconstruction matches the non-Metal baseline closely.
   `ref_post` is nearly as close as raw Metal:
   - metal delta: `[+0.0269, -0.0254]`
   - ref_post delta: `[+0.0239, -0.0176]`

4. Best-offset scanning did not improve over the normal post-RoPE path.
   `ref_post`, `ref_best_metal`, and `ref_best_nonmetal` are identical in this
   run.

5. The remaining discrepancy looks small and second-order.
   The current result does not show a large CoRe distortion caused by the Metal
   capture path.

## Interpretation

The current evidence supports:

- the stable capture boundary is acceptable
- RoPE must be applied
- the remaining Metal vs non-Metal gap is small on this example

The current evidence does not support:

- a large all-token CoRe failure caused by missing RoPE
- a large offset error
- a dramatic analyzer-level mismatch between Metal and non-Metal on this
  matched Granite 4 micro example

## Where To Go Next

1. Run the same repeatable comparison on more CoRe examples.
   One example is not enough to rule out prompt-sensitive failures.

2. Add layer ablations.
   Run the probe with one layer at a time to see which layer contributes most
   to the residual `~0.02` score gap.

3. Add `q`-only and `k`-only substitution variants.
   That will tell us whether the remaining gap is driven mostly by `q` drift or
   `k` drift.

4. Replace the temporary Granite 4 micro layer/head config with a real one.
   The current Granite 4 micro config was adapted from older models and should
   be re-derived properly.

## Artifacts

- Summary JSON:
  [analyzer_score_probe_layers_9_12_15_16_18_sample_0.json](/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/analyzer_score_probe_layers_9_12_15_16_18_sample_0.json)
- Metal bundle:
  [qkv.pt](/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/metal_bundle_corer/qkv.pt)
- Non-Metal bundle:
  [qk.pt](/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_corer/qk.pt)
