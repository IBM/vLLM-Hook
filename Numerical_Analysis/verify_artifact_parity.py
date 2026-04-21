import os
import sys
import tempfile
import shutil
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vllm_hook_plugins"))

MODEL_ID = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2-1.5B-Instruct"
    "/snapshots/ba1cf1846d7df0a0591d6c00649f57e798519da8"
)
DOWNLOAD_DIR = os.path.expanduser("~/.cache")
LAYERS = list(range(1, 21))  # 1-based: layer N = output after Nth transformer block
TARGET_PROMPT_LEN = 64
N_PROMPTS = 8

sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))
from benchmark_hidden_states import _prompts_for_length

def _build_prompts():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=DOWNLOAD_DIR)
    return _prompts_for_length(TARGET_PROMPT_LEN, tokenizer, n=N_PROMPTS)

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["VLLM_HOOK_USE_SAFETENSORS"] = "1"


def run_hook(hook_dir, prompts):
    from vllm_hook_plugins import register_plugins
    from vllm_hook_plugins.hook_llm import HookLLM
    from vllm import SamplingParams
    import tempfile

    register_plugins()

    # Write a temp config with all 20 layers and all_tokens mode
    cfg_src = str(PROJECT_ROOT / "model_configs" / "hidden_states" / "Qwen2-1.5B-Instruct.json")
    import json as _json
    with open(cfg_src) as f:
        cfg = _json.load(f)
    cfg["hidden_states"]["layers"] = LAYERS
    cfg["hidden_states"]["mode"] = "all_tokens"
    tmp_cfg = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
    _json.dump(cfg, tmp_cfg)
    tmp_cfg.close()

    try:
        llm = HookLLM(
            model=MODEL_ID,
            worker_name="probe_hidden_states",
            analyzer_name="hidden_states",
            config_file=tmp_cfg.name,
            download_dir=DOWNLOAD_DIR,
            dtype=torch.float16,
            enable_hook=True,
            enforce_eager=True,
            enable_prefix_caching=False,
            hook_dir=hook_dir,
        )
        params = SamplingParams(temperature=0.0, max_tokens=1)

        # llm.generate() handles hook setup/cleanup and records output req order.
        # analyze() then reorders tensors to match input order automatically.
        outputs = llm.generate(prompts, params)
        stats = llm.analyze(analyzer_spec={"reduce": "none"})

        hs = stats.get("hidden_states", {})
        result = {}
        for layer_name, tensors in hs.items():
            idx = int(layer_name.split(".layers.")[1].split(".")[0])
            result[idx + 1] = [t.cpu().float() for t in tensors]

        run_id_file = llm._run_id_file
        run_id = open(run_id_file).read().strip().split("\n")[-1] if os.path.exists(run_id_file) else None
        art_path = os.path.join(hook_dir, run_id, "tp_rank_0", "hidden_states.safetensors") if run_id else ""
        art_kb = os.path.getsize(art_path) / 1024 if art_path and os.path.exists(art_path) else 0.0

        del llm
        import gc; gc.collect(); torch.cuda.empty_cache()
        return result, art_kb
    finally:
        os.unlink(tmp_cfg.name)


def run_native(prompts):
    import safetensors.torch
    from vllm import LLM, SamplingParams

    tmpdir = tempfile.mkdtemp(prefix="vllm_hs_native_")
    try:
        llm = LLM(
            model=MODEL_ID,
            speculative_config={
                "method": "extract_hidden_states",
                "num_speculative_tokens": 1,
                "draft_model_config": {
                    "hf_config": {
                        "eagle_aux_hidden_state_layer_ids": LAYERS,
                    }
                },
            },
            kv_transfer_config={
                "kv_connector": "ExampleHiddenStatesConnector",
                "kv_role": "kv_producer",
                "kv_connector_extra_config": {"shared_storage_path": tmpdir},
            },
            dtype="float16",
            enforce_eager=True,
            enable_prefix_caching=False,
            download_dir=DOWNLOAD_DIR,
        )
        params = SamplingParams(max_tokens=1)
        outputs = llm.generate(prompts, params)

        # outputs is already in input order (vLLM re-sorts before returning).
        # Each output has one .safetensors file with shape [seq_len, n_layers, hidden_size].
        result = {}  # layer_idx -> list of (seq_len, hidden_size) per prompt, input-ordered
        for out in outputs:
            path = out.kv_transfer_params["hidden_states_path"]
            with safetensors.torch.safe_open(path, framework="pt") as f:
                hs = f.get_tensor("hidden_states").float()  # [seq_len, n_layers, hidden_size]
            for li, layer_idx in enumerate(LAYERS):
                result.setdefault(layer_idx, []).append(hs[:, li, :])  # [seq_len, hidden_size]

        art_kb = sum(
            os.path.getsize(out.kv_transfer_params["hidden_states_path"]) / 1024
            for out in outputs
        )

        del llm
        import gc; gc.collect(); torch.cuda.empty_cache()
        return result, art_kb
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def compare(hook_result, hook_kb, native_result, native_kb):
    print(f"\n{'='*72}")
    print(f"{'Layer':>6}  {'Prompt':>8}  {'Shape':>16}  {'MaxAbsDiff':>12}  {'CosSim':>8}")
    print(f"{'-'*72}")

    for layer_idx in sorted(LAYERS):
        hook_tensors   = hook_result.get(layer_idx, [])
        native_tensors = native_result.get(layer_idx, [])

        if not hook_tensors or not native_tensors:
            print(f"{layer_idx:>6}  {'N/A':>8}  {'missing data':>16}")
            continue

        for pi, (h, n) in enumerate(zip(hook_tensors, native_tensors)):
            min_len = min(h.shape[0], n.shape[0])
            h = h[-min_len:]
            n = n[:min_len]

            max_diff = (h - n).abs().max().item()
            cos_sim  = torch.nn.functional.cosine_similarity(
                h.reshape(1, -1), n.reshape(1, -1)
            ).item()
            print(f"{layer_idx:>6}  {pi:>8}  {str(tuple(h.shape)):>16}  "
                  f"{max_diff:>12.6f}  {cos_sim:>8.6f}")

    print(f"{'='*72}")
    print(f"\nArtifact sizes:")
    print(f"  vLLM-Hook (all_tokens): {hook_kb:.1f} KB")
    print(f"  Native vLLM Eagle:      {native_kb:.1f} KB")
    print(f"  Ratio (native/hook):    {native_kb/hook_kb:.2f}x" if hook_kb > 0 else "")


if __name__ == "__main__":
    prompts = _build_prompts()
    print(f"Using {len(prompts)} prompts of ~{TARGET_PROMPT_LEN} tokens each")

    hook_dir = "/dev/shm/vllm_hook_verify"
    os.makedirs(hook_dir, exist_ok=True)
    try:
        print("\nRunning vLLM-Hook (all_tokens, 20 layers)...")
        hook_result, hook_kb = run_hook(hook_dir, prompts)

        print("\nRunning Native vLLM Eagle (20 layers)...")
        native_result, native_kb = run_native(prompts)

        compare(hook_result, hook_kb, native_result, native_kb)
    finally:
        shutil.rmtree(hook_dir, ignore_errors=True)
