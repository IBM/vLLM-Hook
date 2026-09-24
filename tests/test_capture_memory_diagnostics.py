"""CPU checks for the capture-memory diagnostics.

Run directly with Python or through the repository's pytest suite.
No GPU and no model download required.
"""
import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_hook_plugins._hook_plugin import _log_capture_budget, _looks_like_oom


def _fake_config(layers=28, hidden=1536, heads=12, kv_heads=2, head_dim=128,
                 tokens=8192, gmu=0.78, itemsize=2):
    """Minimal stand-in for VllmConfig with only the fields the budget uses."""
    text = types.SimpleNamespace(
        num_hidden_layers=layers, hidden_size=hidden,
        num_attention_heads=heads, num_key_value_heads=kv_heads,
        head_dim=head_dim,
    )
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(
            hf_text_config=text,
            dtype=types.SimpleNamespace(itemsize=itemsize),
        ),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=tokens),
        cache_config=types.SimpleNamespace(gpu_memory_utilization=gmu),
    )


class _FakeCuda:
    """Stands in for torch.cuda with a fixed device size and free amount."""
    def __init__(self, total=5.64 * 1024**3, free=None):
        self._total, self._free = total, free

    def is_available(self):
        return True

    def get_device_properties(self, _index):
        return types.SimpleNamespace(total_memory=self._total)

    def mem_get_info(self):
        return (self._free, self._total)


def _with_cuda(cuda):
    """Patch the torch module that _log_capture_budget and _looks_like_oom import."""
    return patch.dict(sys.modules, {"torch": types.SimpleNamespace(cuda=cuda)})


class TestCaptureBudget(unittest.TestCase):
    def test_hidden_states_arithmetic(self):
        # 8192 tokens x 28 layers x 1536 x 2 bytes = 672 MiB
        with _with_cuda(_FakeCuda()), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _log_capture_budget(_fake_config(), "hidden_states")
        self.assertEqual(len(caught), 1)
        msg = str(caught[0].message)
        self.assertIn("672 MiB", msg)
        self.assertIn("8192 tokens x 28 layers x 1536", msg)

    def test_qk_worker_uses_projection_widths(self):
        # (12 + 2) heads x 128 head_dim = 1792 per token, not hidden_size
        with _with_cuda(_FakeCuda()), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _log_capture_budget(_fake_config(), "qk")
        self.assertIn("x 1792 x", str(caught[0].message))

    def test_headroom_tracks_gpu_memory_utilization(self):
        sizes = {}
        for gmu in (0.70, 0.90):
            with _with_cuda(_FakeCuda()), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _log_capture_budget(_fake_config(gmu=gmu), "hidden_states")
            sizes[gmu] = str(caught[0].message)
        # Higher utilization must report less free memory.
        self.assertIn("gpu_memory_utilization=0.70", sizes[0.70])
        self.assertIn("gpu_memory_utilization=0.90", sizes[0.90])

    def test_silent_without_cuda(self):
        cuda = _FakeCuda()
        cuda.is_available = lambda: False
        with _with_cuda(cuda), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _log_capture_budget(_fake_config(), "hidden_states")
        self.assertEqual(caught, [])

    def test_malformed_config_does_not_raise(self):
        """A diagnostic must never block engine startup."""
        with _with_cuda(_FakeCuda()), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _log_capture_budget(types.SimpleNamespace(), "hidden_states")
        self.assertEqual(caught, [])


class TestLooksLikeOom(unittest.TestCase):
    def test_explicit_oom_message(self):
        self.assertTrue(_looks_like_oom(RuntimeError("CUDA out of memory. Tried to allocate 132.00 MiB")))

    def test_unrelated_error_is_not_oom(self):
        self.assertFalse(_looks_like_oom(ValueError("bad shape")))

    def test_engine_death_with_exhausted_device(self):
        """vLLM V1 hides the subprocess cause, so fall back to the device."""
        exc = type("EngineDeadError", (Exception,), {})("EngineCore encountered an issue.")
        with _with_cuda(_FakeCuda(free=106 * 1024**2)):   # 1.8% free, as observed
            self.assertTrue(_looks_like_oom(exc))

    def test_engine_death_with_free_device_is_not_oom(self):
        exc = type("EngineDeadError", (Exception,), {})("EngineCore encountered an issue.")
        with _with_cuda(_FakeCuda(free=5.54 * 1024**3)):  # idle GPU
            self.assertFalse(_looks_like_oom(exc))

    def test_device_query_failure_is_not_oom(self):
        exc = type("EngineDeadError", (Exception,), {})("EngineCore encountered an issue.")
        cuda = _FakeCuda()
        cuda.mem_get_info = lambda: (_ for _ in ()).throw(RuntimeError("no device"))
        with _with_cuda(cuda):
            self.assertFalse(_looks_like_oom(exc))


if __name__ == "__main__":
    unittest.main(verbosity=2)
