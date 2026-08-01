"""Repo-layout shim.

The installable package lives one level down, beside ``setup.py``. When
the repository root is on ``sys.path`` this outer directory would shadow
the installed package and break its absolute imports, so it replaces
itself with the inner package in ``sys.modules`` — repo-root imports then
behave exactly like installed ones.
"""
import sys

from . import vllm_hook_plugins as _package

sys.modules[__name__] = _package
