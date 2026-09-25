from setuptools import setup, find_packages

setup(
    name="mia",
    version="0.6.0",
    packages=find_packages(include=["mia", "mia.*"]),
    # PINNED: MIA targets vLLM 0.29.0 and its V2 model runner exclusively. Older
    # vLLM has no V2 runner; newer releases move V2 internals (prepare_inputs already
    # changed signature between 0.25 and 0.29), so a range would be a silent-breakage
    # promise we cannot keep. Re-validate, then move the pin.
    install_requires=["vllm==0.29.0", "zstandard"],
    entry_points={
        "vllm.general_plugins": [
            "mia_registry = mia:register_plugins",
            "mia = mia._plugin:register",
        ],
    },
    python_requires=">=3.10",
)
