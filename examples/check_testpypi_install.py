"""Smoke test for the vllm-hook package installed from TestPyPI.

Full test (Linux + NVIDIA GPU, Python 3.11-3.14; vllm 0.32.0 has no macOS wheel):
    python -m venv /tmp/mia-test && source /tmp/mia-test/bin/activate
    pip install vllm==0.32.0 zstandard      # dependencies from real PyPI
    pip install --no-deps --index-url https://test.pypi.org/simple/ vllm-hook==1.0.1a0

    Do not use --extra-index-url for the dependencies: TestPyPI hosts junk copies of popular
    packages (e.g. a jmespath that fails to build) and pip may pick them over the PyPI ones.

Packaging-only test (any OS, e.g. a Mac; skips the import because vllm is missing):
    pip install --no-deps --index-url https://test.pypi.org/simple/ vllm-hook==1.0.1a0

Run from any directory outside the repo (so the local `mia/` is not imported):
    python check_testpypi_install.py            # install checks (no GPU)
    python check_testpypi_install.py --run      # also capture hidden states (needs a GPU)
"""
import importlib.util
import sys
from importlib.metadata import entry_points, files, requires, version


def check_metadata():
    """Checks that need no vllm: version, dependencies, entry points, shipped data files."""
    print("vllm-hook", version("vllm-hook"), "| requires:", requires("vllm-hook"))
    eps = [e.name for e in entry_points(group="vllm.general_plugins")]
    assert "mia" in eps and "mia_registry" in eps, f"plugin entry points missing: {eps}"
    print("vLLM plugin entry points:", eps)

    names = [str(f) for f in files("vllm-hook")]
    for needle in ("mia/resources.py", "mia/data/model_configs/hidden_states/Qwen2-1.5B-Instruct.json",
                   "mia/data/steering_vectors/phi3_format.pt"):
        assert any(n.endswith(needle) for n in names), f"missing from wheel: {needle}"
    print("wheel ships mia/ plus", sum("mia/data/" in n for n in names), "config/vector files")


def check_install():
    import mia

    print("vllm-hook", version("vllm-hook"), "from", mia.__file__)
    assert "site-packages" in mia.__file__, "imported a local checkout, not the installed package"

    eps = [e.name for e in entry_points(group="vllm.general_plugins")]
    assert "mia" in eps and "mia_registry" in eps, f"plugin entry points missing: {eps}"
    print("vLLM plugin entry points:", eps)

    assert mia.config_path("hidden_states/Qwen2-1.5B-Instruct.json").endswith(".json")
    assert mia.vector_path("phi3_format.pt").endswith(".pt")
    print("packaged configs and steering vectors found")
    mia.register_plugins()
    print("registry OK; public API:", ", ".join(mia.__all__[:4]), "...")


def check_run():
    import multiprocessing as mp
    import os

    from vllm import SamplingParams

    from mia import MiaLLM, config_path

    mp.set_start_method("spawn", force=True)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    config_file = config_path("hidden_states/Qwen2-1.5B-Instruct.json")  # shipped in the wheel
    print("config:", config_file)
    llm = MiaLLM(model="Qwen/Qwen2-1.5B-Instruct", worker_name="capture_hs",
                 analyzer_name="hidden_states", config_file=config_file,
                 gpu_memory_utilization=0.7, max_model_len=2048)
    out = llm.generate("The capital of France is", SamplingParams(temperature=0.0, max_tokens=10))
    print("generated:", out[0].outputs[0].text)
    stats = llm.analyze(analyzer_spec={"reduce": "none"}, probes=out[0].probes)
    for layer, tensors in sorted(stats["hidden_states"].items()):
        print("layer", layer, tuple(tensors[0].shape))
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    check_metadata()
    if importlib.util.find_spec("vllm") is None:
        print("vllm not installed: skipped import and run checks (packaging checks passed)")
        sys.exit(0)
    check_install()
    if "--run" in sys.argv:
        check_run()
    print("OK")
