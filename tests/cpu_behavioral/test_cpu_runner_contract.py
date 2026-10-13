"""Document the vLLM CPU runner boundary for MIA CPU parity work."""
from __future__ import annotations

import importlib

import pytest


pytestmark = pytest.mark.cpu_behavior


def test_vllm_cpu_runner_is_gpu_runner_subclass():
    cpu_mod = pytest.importorskip("vllm.v1.worker.cpu.model_runner")
    gpu_mod = pytest.importorskip("vllm.v1.worker.gpu.model_runner")

    assert issubclass(cpu_mod.CPUModelRunner, gpu_mod.GPUModelRunner)


def test_mia_runner_guard_accepts_cpu_runner_for_parity_experiment():
    cpu_mod = pytest.importorskip("vllm.v1.worker.cpu.model_runner")
    runner_mod = importlib.import_module("mia.core.runner")

    assert runner_mod.is_v2_runner(cpu_mod.CPUModelRunner.__new__(cpu_mod.CPUModelRunner)) is True
