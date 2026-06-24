from setuptools import setup

setup(
    name="live-highlighter",
    version="0.1.0",
    packages=["live_highlighter"],
    package_dir={"live_highlighter": "."},
    package_data={"live_highlighter": ["assets/*"]},
    include_package_data=True,
    install_requires=["ipywidgets"],
    python_requires=">=3.8",
)
