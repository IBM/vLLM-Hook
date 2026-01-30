from setuptools import setup, find_packages

setup(
    name="vllm-hook-plugins",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["vllm"],
    entry_points={
        "vllm.general_plugins": [
            "hook_registry = vllm_hook_plugins:register_plugins",
        ],
    },
    python_requires=">=3.8",
)