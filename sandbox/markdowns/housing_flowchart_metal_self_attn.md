# Flow Chart For The Current Metal Worker

This flow chart matches the current `self_attn`-wrapper Metal worker and uses the same housing analogy as [housing_analogy_metal_self_attn.md](/Users/timothyburley/opensource/vLLM-Hook/sandbox/markdowns/housing_analogy_metal_self_attn.md).

## Compact View

```mermaid
flowchart TD
    A[HookLLM creates run ID<br/>and hook flag] --> B[Metal worker loads model]
    B --> C[Find tracked<br/>self_attn houses]
    C --> D[Replace each with<br/>wrapper outer house]
    D --> E[execute_model starts<br/>capture active]
    E --> F[Visitor packet x enters<br/>self_attn door]
    F --> G[Wrapper recording room<br/>reads run ID]
    G --> H{Hook flag present<br/>and capture active?}
    H -- No --> I[Forward visitor to<br/>original inner house]
    H -- Yes --> J[Compute q_proj x<br/>k_proj x<br/>v_proj x]
    J --> K[Reshape and transpose<br/>Q K V packets]
    K --> L[Apply rope to Q and K<br/>using cache offset if present]
    L --> M[Store x q k v packets<br/>in run notebook]
    M --> I
    I --> N[Original self_attn house<br/>runs normally]
    N --> O[execute_model ends]
    O --> P[Flush notebook to<br/>qkv.pt archive]
    P --> Q[run_utils normalizes<br/>qkv to qk view]
    Q --> R[Analyzer scores<br/>tracked heads]
```

## Housing View

```mermaid
flowchart TD
    A[Building directory<br/>named_modules] --> B[Find tracked floors]
    B --> C[At each self_attn address<br/>install wrapper outer house]
    C --> D[Visitor arrives at<br/>outer house door]
    D --> E{Recording room open?}
    E -- No --> F[Send visitor straight to<br/>inner house]
    E -- Yes --> G[Write down raw visitor<br/>packet x]
    G --> H[Make three specialist copies:<br/>Q visitor K visitor V visitor]
    H --> I[Apply house reshaping rules]
    I --> J[Send Q and K through<br/>position checkpoint rope]
    J --> K[File x q k v into notebook<br/>for this run and layer]
    K --> F
    F --> L[Inner house performs<br/>real attention work]
    L --> M[Notebook archived as<br/>qkv.pt]
    M --> N[Scoring desk reads notes<br/>through normalized qk view]
```

## Short Reading Guide

- `wrapper outer house`
  - the installed `MLXHookWrapper` at `model.layers.<i>.self_attn`
- `inner house`
  - the original Metal Granite `Attention` module
- `visitor packet x`
  - the raw hidden-state tensor entering `self_attn`
- `notebook`
  - `self._run_cache`
- `archive`
  - `qkv.pt`
- `scoring desk`
  - the analyzer path after `run_utils` normalizes `qkv_cache` into `qk_cache`
