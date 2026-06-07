from vllm_hook_plugins.metal.hook_llm_metal import HookLLMMetal


def test_metal_hidden_states_worker_resolves_to_metal_impl():
    path = HookLLMMetal._resolve_worker_class_path("probe_hidden_states")

    assert path is not None
    assert path.endswith("ProbeHiddenStatesWorkerMetal")
    assert "workers.metal.probe_hidden_states_worker_metal" in path


def test_metal_hidden_states_worker_alias_resolves_to_metal_impl():
    path = HookLLMMetal._resolve_worker_class_path("probe_hidden_states_metal")

    assert path is not None
    assert path.endswith("ProbeHiddenStatesWorkerMetal")
    assert "workers.metal.probe_hidden_states_worker_metal" in path


def test_metal_hidden_states_analyzer_resolves_to_metal_impl():
    analyzer_cls = HookLLMMetal._resolve_analyzer_class("hidden_states")

    assert analyzer_cls.__name__ == "HiddenStatesAnalyzerMetal"
    assert "analyzers.metal.hidden_states_analyzer_metal" in analyzer_cls.__module__


def test_metal_hidden_states_analyzer_alias_resolves_to_metal_impl():
    analyzer_cls = HookLLMMetal._resolve_analyzer_class("hidden_states_metal")

    assert analyzer_cls.__name__ == "HiddenStatesAnalyzerMetal"
    assert "analyzers.metal.hidden_states_analyzer_metal" in analyzer_cls.__module__


def test_metal_hidden_states_indexed_fallback_installs_inside_layer_loop():
    import ast
    from pathlib import Path

    source = Path(
        "vllm_hook_plugins/vllm_hook_plugins/workers/metal/"
        "probe_hidden_states_worker_metal.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    install_method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_install_hooks"
    )
    indexed_loop = next(
        node
        for node in ast.walk(install_method)
        if isinstance(node, ast.For)
        and isinstance(node.iter, ast.Call)
        and getattr(node.iter.func, "id", "") == "enumerate"
        and getattr(node.iter.args[0], "id", "") == "indexed_layers"
    )

    assert any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "install_wrapper"
        for node in ast.walk(indexed_loop)
    )
