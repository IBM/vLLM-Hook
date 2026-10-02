"""Package metadata and vLLM plugin entry points for MIA."""
from setuptools import setup, find_packages

setup(
    name="mia",
    # MIA v0. Deliberately not a release number: this contribution does not claim one, and
    # what the package version should say upstream is the maintainers' call. ("MIA_v0" itself
    # is not a legal PEP 440 version, so the field carries the part that has to be machine
    # readable and the name carries the rest.)
    version="0",
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

