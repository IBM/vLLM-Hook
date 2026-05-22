from setuptools import setup, find_packages

setup(
    name="vllm-hook-plugins",
    version="0.2.0",
    packages=find_packages(),
    install_requires=["vllm", "zstandard"],
    entry_points={
        "vllm.general_plugins": [
            "hook_registry = vllm_hook_plugins:register_plugins",
            "vllm_hook = vllm_hook_plugins._hook_plugin:register",
        ],
    },
    python_requires=">=3.8",
)