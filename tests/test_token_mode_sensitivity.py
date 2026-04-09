import importlib.util
import sys
import types
from pathlib import Path

import torch


def _load_analyzer_classes():
    plugin_root = (
        Path(__file__).resolve().parents[1]
        / "vllm_hook_plugins"
        / "vllm_hook_plugins"
    )

    pkg = types.ModuleType("vllm_hook_plugins")
    pkg.__path__ = [str(plugin_root)]
    sys.modules.setdefault("vllm_hook_plugins", pkg)

    run_utils_spec = importlib.util.spec_from_file_location(
        "vllm_hook_plugins.run_utils",
        plugin_root / "run_utils.py",
    )
    run_utils = importlib.util.module_from_spec(run_utils_spec)
    sys.modules["vllm_hook_plugins.run_utils"] = run_utils
    run_utils_spec.loader.exec_module(run_utils)

    attn_spec = importlib.util.spec_from_file_location(
        "test_attention_tracker_analyzer",
        plugin_root / "analyzers" / "attention_tracker_analyzer.py",
    )
    attn_module = importlib.util.module_from_spec(attn_spec)
    attn_spec.loader.exec_module(attn_module)

    corer_spec = importlib.util.spec_from_file_location(
        "test_core_reranker_analyzer",
        plugin_root / "analyzers" / "core_reranker_analyzer.py",
    )
    corer_module = importlib.util.module_from_spec(corer_spec)
    corer_spec.loader.exec_module(corer_module)

    return attn_module.AttntrackerAnalyzer, corer_module.CorerAnalyzer


AttntrackerAnalyzer, CorerAnalyzer = _load_analyzer_classes()


def _write_qk_cache(hook_dir, run_id, q_tensor, k_tensor):
    run_dir = hook_dir / run_id / "tp_rank_0"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "hidden_size": 2,
                "head_dim": 1,
                "attention_multiplier": 1.0,
            },
            "qk_cache": {
                "model.layers.0.self_attn.attn": {
                    "q": [q_tensor],
                    "k_all": [k_tensor],
                    "layer_num": 0,
                }
            },
        },
        run_dir / "qk.pt",
    )


def test_attention_tracker_ignores_non_last_query_token_changes(tmp_path):
    hook_dir = tmp_path / "hook"
    run_a = "run_a"
    run_b = "run_b"
    run_ids = tmp_path / "run_ids.txt"
    analyzer = AttntrackerAnalyzer(str(hook_dir), {0: [0, 1]})

    # Only the first query token changes. The last token is identical.
    q_a = torch.tensor([[8.0, 1.0], [0.0, 6.0]])
    q_b = torch.tensor([[-3.0, 4.0], [0.0, 6.0]])
    k_all = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    _write_qk_cache(hook_dir, run_a, q_a, k_all)
    _write_qk_cache(hook_dir, run_b, q_b, k_all)

    run_ids.write_text(f"{run_a}\n")
    attn_a = analyzer.compute_attention_from_qk(str(run_ids))
    run_ids.write_text(f"{run_b}\n")
    attn_b = analyzer.compute_attention_from_qk(str(run_ids))

    attn_tensor_a = attn_a[0]["model.layers.0.self_attn.attn"]["attention"]
    attn_tensor_b = attn_b[0]["model.layers.0.self_attn.attn"]["attention"]
    assert torch.allclose(attn_tensor_a, attn_tensor_b)


def test_core_reranker_changes_when_earlier_query_tokens_change(tmp_path):
    hook_dir = tmp_path / "hook"
    analyzer = CorerAnalyzer(str(hook_dir), {0: [0, 1]})

    # Last query token is identical across both runs.
    # Earlier query token differs and should change the all-token reranker score.
    # The changed token sits in the middle so the causal mask still allows it
    # to distribute attention across multiple earlier keys.
    q_a = torch.tensor([[0.0, 0.0], [8.0, 0.0], [0.0, 6.0]])
    q_b = torch.tensor([[0.0, 0.0], [-8.0, 0.0], [0.0, 6.0]])
    k_all = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    _write_qk_cache(hook_dir, "run_a", q_a, k_all)
    _write_qk_cache(hook_dir, "run_b", q_b, k_all)

    doc_span = [[(0, 0), (1, 1)]]
    query_start = [0]
    after_instruct = [0]
    query_end = [2]

    scores_a, _ = analyzer.score_documents(
        "run_a",
        doc_span,
        query_start,
        after_instruct,
        query_end,
    )
    scores_b, _ = analyzer.score_documents(
        "run_b",
        doc_span,
        query_start,
        after_instruct,
        query_end,
    )

    assert torch.allclose(q_a[-1], q_b[-1])
    assert not torch.allclose(scores_a[0][0], scores_b[0][0])
