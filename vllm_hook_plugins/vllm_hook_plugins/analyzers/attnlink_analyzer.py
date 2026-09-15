"""AttnLink-U: rank schema columns using one generation-anchor attention head."""
import math
from typing import Dict, Optional

import torch

from vllm_hook_plugins.run_utils import load_and_merge_qk_cache, unpack_qk


def select_columns(scores, temperature=1.0, top_p=0.8):
    """Paper Eq. (5), (13): temperature-scaled candidate mass and its top-p prefix.

    No gold labels or top-k limit are used. Returned arrays/indices refer to the
    original candidate order. top_p=1 retains every candidate.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive.")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1].")
    values = torch.tensor(scores, dtype=torch.float64)
    if values.ndim != 1 or not values.numel() or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Scores must be a nonempty list of finite nonnegative values.")
    logits = torch.log(values + 1e-8)
    # Center before dividing for stability even at very small temperatures.
    probabilities = torch.softmax((logits - logits.max()) / temperature, dim=0).tolist()
    ranking = sorted(range(len(scores)), key=lambda i: -scores[i])
    selected, cumulative = [], 0.0
    for i in ranking:
        selected.append(i)
        cumulative += probabilities[i]
        if top_p < 1 and cumulative >= top_p:
            break
    return {"probabilities": probabilities, "ranking": ranking, "selected": selected}


class AttnLinkAnalyzer:
    def __init__(self, hook_dir: str, layer_to_heads: Dict[int, list]):
        self.hook_dir = hook_dir
        if len(layer_to_heads) != 1 or len(next(iter(layer_to_heads.values()))) != 1:
            raise ValueError("AttnLink requires exactly one configured layer/head pair.")
        self.layer, heads = next(iter(layer_to_heads.items()))
        self.head = heads[0]

    def analyze(self, analyzer_spec: Dict, run_id: Optional[str] = None,
                probes: Optional[Dict] = None) -> Dict:
        """Score a single prompt; gold labels are deliberately not part of this API.

        analyzer_spec contains candidates (identifiers), candidate_spans
        (half-open token ranges), and prompt_length. Capture must use last_token,
        one GPU, and a complete, unchunked prefill without prefix caching.
        Optional temperature (default 1.0) and top_p (default 0.8) control column
        selection. Scores/probabilities follow candidate order; ranking and
        selected contain indices. Gold labels are used only by the demo.
        """
        candidates = analyzer_spec["candidates"]
        spans = analyzer_spec["candidate_spans"]
        length = analyzer_spec["prompt_length"]
        if not candidates or len(candidates) != len(spans):
            raise ValueError("Provide one nonempty span per candidate.")
        if len(set(candidates)) != len(candidates):
            raise ValueError("Candidate identifiers must be unique.")
        if length <= 0 or any(not 0 <= start < end <= length for start, end in spans):
            raise ValueError("Candidate spans must be nonempty and within the prompt.")
        if probes is None:
            if run_id is None:
                raise ValueError("Pass QK probes or a disk run_id.")
            probes = load_and_merge_qk_cache(self.hook_dir, run_id)
        layers = [entry for entry in probes.get("qk_cache", {}).values()
                  if entry["layer_num"] == self.layer]
        if len(layers) != 1:
            raise ValueError(f"Expected QK capture for layer {self.layer}.")
        entry = layers[0]
        if entry.get("hookq_mode") != "last_token":
            raise ValueError("Capture must use hookq_mode=last_token.")
        queries, keys = unpack_qk(entry)
        if len(queries) != 1 or len(keys) != 1:
            raise ValueError("AttnLink demo expects one prompt and one prefill pass.")
        q, k = queries[0], keys[0]
        config = probes["config"]
        heads = config["num_attention_heads"]
        kv_heads = config["num_key_value_heads"]
        dim = config["head_dim"]
        if heads <= 0 or kv_heads <= 0 or dim <= 0 or heads % kv_heads:
            raise ValueError("Invalid GQA head dimensions.")
        if not 0 <= self.head < heads:
            raise ValueError("Configured attention head is out of range.")
        if q.shape != (heads * dim,) or k.shape != (length, kv_heads * dim):
            raise ValueError("QK dimensions do not match a full single-GPU prompt capture.")
        # GQA: consecutive groups of query heads share one key head.
        kv_head = self.head // (heads // kv_heads)
        query = q.reshape(heads, dim)[self.head].float()
        key = k.reshape(length, kv_heads, dim)[:, kv_head].float()
        if not torch.isfinite(query).all() or not torch.isfinite(key).all():
            raise ValueError("Captured QK contains nonfinite values.")
        scale = config.get("attention_multiplier", 1.0 / math.sqrt(dim))
        # The final prompt position can attend to every prompt token. Normalize
        # over the entire prompt before pooling, not over candidate tokens only.
        attention = torch.softmax(torch.mv(key, query) * scale, dim=0)
        scores = [attention[start:end].mean().item() for start, end in spans]
        selection = select_columns(scores, analyzer_spec.get("temperature", 1.0),
                                   analyzer_spec.get("top_p", 0.8))
        return {"scores": scores, **selection}
