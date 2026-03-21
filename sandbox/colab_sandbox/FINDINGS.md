# CoRe Granite 4 Micro Findings

This note summarizes the current matched Metal vs non-Metal CoRe comparison
using the repeatable `sandbox/colab_sandbox` workflow.

## Prompt Used

The matched CoRe comparison used the checked-in input payload from
[corer_input.json](/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/corer_input.json):

Query:

```text
Which magazine was started first Arthur's Magazine or First for Women?
```

Documents:

1. `Arthur's Magazine was an American literary periodical published in the 1840s.`
2. `First for Women is a woman's magazine published in the USA and was started in 1989.`

NA query:

```text
N/A
```

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

Per-document token mass from the probe output:

- `nonmetal` doc 1 mass is concentrated mostly in the last two positions:
  - `0.6611`
  - `0.1155`
- `nonmetal` doc 2 mass is concentrated mostly in the last two positions:
  - `1.1328`
  - `0.8696`
- `metal` preserves that overall pattern:
  - doc 1 tail mass: `0.6846`, `0.1147`
  - doc 2 tail mass: `1.1179`, `0.8601`

That means the Metal residual is not coming from a completely different
document-selection pattern. It is a small redistribution of mass around the
same high-impact document tokens.

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

6. Layer-level tensor drift is small, consistent, and largest in deeper layers.
   From the probe JSON:

   - layer 12:
     - Metal vs non-Metal `q` mean abs diff: `0.02329`
     - Metal vs non-Metal `k` mean abs diff: `0.02835`
   - layer 15:
     - `q`: `0.02568`
     - `k`: `0.03079`
   - layer 16:
     - `q`: `0.02509`
     - `k`: `0.03176`
   - layer 18:
     - `q`: `0.03013`
     - `k`: `0.04179`

   Cosine similarity remains extremely high across all of these comparisons:
   roughly `0.99996` to `0.99998`.

7. `k` drift is consistently larger than `q` drift.
   That suggests the residual CoRe score gap may be more sensitive to small key
   differences than to query differences in this example.

8. `ref_post` is slightly closer to the non-Metal tensors than raw Metal.
   Example:

   - layer 12:
     - Metal `q`: `0.02329`
     - Post-RoPE reference `q`: `0.02217`
     - Metal `k`: `0.02835`
     - Post-RoPE reference `k`: `0.02677`
   - layer 18:
     - Metal `q`: `0.03013`
     - Post-RoPE reference `q`: `0.02920`
     - Metal `k`: `0.04179`
     - Post-RoPE reference `k`: `0.04029`

   This reinforces that the remaining issue is not a gross semantic failure.
   It looks like small numeric/path-specific drift after the correct
   attention-ready reconstruction steps.

## Interpretation

The current evidence supports:

- the stable capture boundary is acceptable
- RoPE must be applied
- the remaining Metal vs non-Metal gap is small on this example

The current evidence also suggests:

- the residual score gap is driven by small tensor drift across multiple layers
  rather than one catastrophic failure
- later layers, especially layer `18`, are the strongest current candidates for
  explaining the remaining difference
- the dominant residual may be in `k` rather than `q`

## Larger-Document Hypothesis

It is fair to treat document length as a plausible risk factor for larger CoRe
drift.

Reasoning:

- CoRe aggregates all-token attention mass over document spans
- the matched Granite 4 micro test already shows small Metal vs non-Metal
  tensor drift across multiple layers
- with longer documents, there are more document-token positions over which
  that small drift can accumulate

So the conservative statement is:

- the current evidence is consistent with the idea that CoRe drift can worsen
  as document length grows
- but the current matched test does not prove that yet

## Why This Hypothesis Is Reasonable

The original larger-document CoRe run showed a much wider Metal vs non-Metal
margin than the small 2-document control example summarized above.

Original Metal result:

```text
Sorted document IDs and scores by CoRe-Reranking:
[[6, 3, 1, 0, 4, 5, 2]]:
[[5.625, 3.921875, 2.296875, 2.046875, 1.53125, 0.734375, 0.54296875]]
```

Original non-Metal result:

```text
Sorted document IDs and scores by CoRe-Reranking:
[[6, 3, 1, 0, 4, 2, 5]]:
[[4.04296875, 3.427734375, 2.419921875, 1.6767578125, 1.62890625, 1.01953125, 0.79736328125]]
```

What that suggests:

- the ranking prefix remains mostly stable
- the score margin is substantially larger than in the matched 2-document test
- the lower-ranked documents can swap order under the accumulated drift

That pattern is consistent with an accumulation story:

- small per-layer tensor drift
- distributed over many more document tokens
- producing larger integrated CoRe score differences

But there is still an important caveat:

- this larger-document example is suggestive, not isolating
- it is not yet a clean length-controlled sweep
- so it supports the hypothesis without proving that length alone is the cause

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

5. Compare token-level CoRe deltas directly.
   The current summary already shows that the largest mass sits on the same
   document-tail tokens for Metal and non-Metal. The next step is to compute
   explicit token-by-token deltas and rank which positions explain the final
   `doc_sums` difference.

6. Run a document-length sweep.
   Keep the same query and relevant evidence, then progressively increase the
   document span length and measure whether the Metal vs non-Metal CoRe gap
   grows monotonically or near-monotonically.

## Artifacts

- Summary JSON:
  [analyzer_score_probe_layers_9_12_15_16_18_sample_0.json](/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/analyzer_score_probe_layers_9_12_15_16_18_sample_0.json)
- Metal bundle:
  [qkv.pt](/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/metal_bundle_corer/qkv.pt)
- Non-Metal bundle:
  [qk.pt](/Users/timothyburley/opensource/vLLM-Hook/sandbox/colab_sandbox/output/nonmetal_bundle_corer/qk.pt)
