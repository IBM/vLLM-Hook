"""Package metadata and vLLM plugin entry points for vLLM-Hook MIA."""
from setuptools import setup, find_packages

setup(
    name="vllm-hook-mia",
    version="0.1.0",
    packages=find_packages(include=["mia", "mia.*"]),
    install_requires=["vllm==0.29.0", "zstandard"],
    entry_points={
        "vllm.general_plugins": [
            "mia_registry = mia:register_plugins",
            "mia = mia.core._plugin:register",
        ],
    },
    python_requires=">=3.11,<3.15",
)
