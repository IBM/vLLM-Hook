from setuptools import setup, find_packages

setup(
    name="vllm-hook-plugins",
    version="0.3.1",
    packages=find_packages(),
    install_requires=["torch>=2.0", "numpy>=1.24", "safetensors", "zstandard"],
    extras_require={
        "engine": ["vllm>=0.9,<=0.21"],
    },
    entry_points={
        "vllm.general_plugins": [
            "hook_registry = vllm_hook_plugins:register_plugins",
            "vllm_hook = vllm_hook_plugins._hook_plugin:register",
        ],
    },
    python_requires=">=3.10",
)