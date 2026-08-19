from setuptools import setup, find_packages

setup(
    name="vllm-hook-plugins",
    version="0.3.0",
    packages=find_packages(),
    install_requires=["torch>=2.0", "numpy>=1.24", "safetensors"],
    extras_require={
        # Engine range: floor = oldest release with worker_extension_cls +
        # collective_rpc + SamplingParams.extra_args + per-request cache_salt;
        # verified in CI, advanced release-by-release.
        "engine": ["vllm>=0.9,<=0.21", "zstandard"],
    },
    entry_points={
        "vllm.general_plugins": [
            "hook_registry = vllm_hook_plugins:register_plugins",
            "vllm_hook = vllm_hook_plugins._hook_plugin:register",
        ],
    },
    python_requires=">=3.10",
)
