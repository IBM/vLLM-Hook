"""H-Node hallucination detection — inference side for vLLM-Hook.

Loads a pre-trained H-Node probe artifact and scores last-token hidden states
to detect hallucination at inference time. Implements the detection half of:

    "H-Node Attack and Defense in Large Language Models"
    Yocam, Vaidyan, Wang, 2026 — https://arxiv.org/abs/2603.26045

To build a probe artifact from scratch (extract activations, train probes,
select H-Nodes), see the config-building repository:

    https://github.com/Samarpit-bhatia/hnode-probe-builder
"""

from vllm_hook_plugins.utils.hnode.score import (
    HNodeProbe,
    ProbeArtifact,
    score_activations,
)

__all__ = ["ProbeArtifact", "HNodeProbe", "score_activations"]
