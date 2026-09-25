"""Package metadata and vLLM plugin entry points for MIA."""
from setuptools import setup, find_packages

setup(
    name="mia",
    version="0.6.0",
    packages=find_packages(include=["mia", "mia.*"]),
    install_requires=["vllm==0.29.0", "zstandard"],
    entry_points={
        "vllm.general_plugins": [
            "mia_registry = mia:register_plugins",
            "mia = mia._plugin:register",
        ],
    },
    python_requires=">=3.10",
)

