"""CPU behavior baseline for hidden-state analyzer outputs.

This does not boot a vLLM engine. It checks the CPU-side analyzer contract that
the MIA hidden-state example depends on: one finite hidden-state tensor per
prompt, per captured layer.
"""
import pytest
import torch

from mia.analyzers.hidden_states_analyzer import HiddenStatesAnalyzer


@pytest.mark.cpu_behavior
def test_hidden_states_analyzer_accepts_cpu_probe_tensors(tmp_path):
    hidden_size = 8
    prompts = ["Hidden states test prompt one.", "Hidden states test prompt two."]
    probes = {
        "hs_cache": {
            "layers.0": {
                "hidden_states": [
                    torch.arange(hidden_size, dtype=torch.float32, device="cpu"),
                    torch.ones(hidden_size, dtype=torch.float32, device="cpu"),
                ],
                "layer_num": 0,
                "hs_mode": "last_token",
            }
        }
    }

    analyzer = HiddenStatesAnalyzer(str(tmp_path), layer_to_heads={})

    stats = analyzer.analyze(analyzer_spec={"reduce": "none"}, probes=probes)

    assert "hidden_states" in stats
    hidden_states = stats["hidden_states"]
    assert list(hidden_states) == ["layers.0"]

    tensors = hidden_states["layers.0"]
    assert len(tensors) == len(prompts)
    for tensor in tensors:
        assert tensor.device.type == "cpu"
        assert tensor.shape == torch.Size([hidden_size])
        assert torch.isfinite(tensor).all()


@pytest.mark.cpu_behavior
def test_hidden_states_analyzer_reduces_cpu_probe_tensors_to_norms(tmp_path):
    probes = {
        "hs_cache": {
            "layers.0": {
                "hidden_states": [
                    torch.tensor([3.0, 4.0], device="cpu"),
                    torch.tensor([5.0, 12.0], device="cpu"),
                ],
                "layer_num": 0,
                "hs_mode": "last_token",
            }
        }
    }

    analyzer = HiddenStatesAnalyzer(str(tmp_path), layer_to_heads={})

    stats = analyzer.analyze(analyzer_spec={"reduce": "norm"}, probes=probes)

    assert stats["hidden_states"]["layers.0"] == [5.0, 13.0]
