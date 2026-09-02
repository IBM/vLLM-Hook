from setuptools import setup, find_packages

setup(
    name="vllm-hook-plugins",
    version="0.4.0",
    packages=find_packages(),
    # zstandard is imported at module load by _hook_plugin and the probe
    # workers, and register_plugins() imports every worker eagerly, so it is
    # a hard dependency of the bare package rather than of the [engine] extra.
    install_requires=["torch>=2.0", "numpy>=1.24", "safetensors", "zstandard"],
    extras_require={
        # Engine range: the workers read the legacy GPU model runner's
        # input_batch / requests surface, and on releases that default to
        # the V2 runner (0.28+) the plugin pins the legacy one via
        # VLLM_USE_V2_MODEL_RUNNER=0 at engine-config time. Aligned with the
        # steerability toolkit's range; verified against 0.28, advanced
        # release-by-release.
        "engine": ["vllm>=0.26,<1.0"],
    },
    entry_points={
        "vllm.general_plugins": [
            "hook_registry = vllm_hook_plugins:register_plugins",
            "vllm_hook = vllm_hook_plugins._hook_plugin:register",
        ],
    },
    python_requires=">=3.10",
)
