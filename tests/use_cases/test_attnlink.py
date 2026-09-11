"""CPU checks; run directly with Python or through the repository's pytest suite."""
import copy
import hashlib
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))
from demo_attnlink import INPUT_SEQ, evaluate_ranking, prepare_prompt
from vllm_hook_plugins.analyzers.attnlink_analyzer import AttnLinkAnalyzer, select_columns


class CharacterTokenizer:
    """Deterministic offsets isolate span bookkeeping from a model download."""
    def apply_chat_template(self, messages, **kwargs):
        return "<user>" + messages[0]["content"] + "</user><assistant>"

    def __call__(self, text, **kwargs):
        return {"input_ids": list(map(ord, text)),
                "offset_mapping": [(i, i + 1) for i in range(len(text))]}


class TestAttnLink(unittest.TestCase):
    def setUp(self):
        q = torch.zeros(8)
        q[6] = 1.0  # query head 3 must use KV head 1, not KV head 0
        k = torch.zeros(5, 2, 2)
        k[:, 0, 0] = 100
        k[:, 1, 0] = torch.tensor([0, math.log(2), math.log(4), 0, math.log(2)])
        self.capture = {
            "config": {"num_attention_heads": 4, "num_key_value_heads": 2,
                       "head_dim": 2, "attention_multiplier": 1.0},
            "qk_cache": {"layer": {"layer_num": 2, "hookq_mode": "last_token",
                                    "q": q[None], "k_all": k.reshape(1, 5, 4)}}}
        self.spec = {"candidates": ["a@t", "b@t"], "candidate_spans": [(1, 3), (4, 5)],
                     "prompt_length": 5}
        self.analyzer = AttnLinkAnalyzer("", {2: [3]})

    def test_gqa_full_softmax_and_mean_pooling(self):
        result = self.analyzer.analyze(self.spec, probes=self.capture)
        # Full probabilities: [0.1, 0.2, 0.4, 0.1, 0.2]. Candidate-only
        # normalization and span-sum pooling both give different results.
        torch.testing.assert_close(torch.tensor(result["scores"]), torch.tensor([0.3, 0.2]))
        self.assertEqual(result["ranking"], [0, 1])

    def test_disk_and_rpc_agree(self):
        disk = copy.deepcopy(self.capture)
        entry = disk["qk_cache"]["layer"]
        entry["q"] = list(entry["q"].unbind(0))
        entry["k_all"] = list(entry["k_all"].unbind(0))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {
                "VLLM_HOOK_USE_SAFETENSORS": "0", "VLLM_HOOK_ASYNC_SAVE": "0"}):
            run = Path(folder) / "test_run"
            run.mkdir()
            torch.save(disk, run / "qk.pt")
            analyzer = AttnLinkAnalyzer(folder, {2: [3]})
            self.assertEqual(analyzer.analyze(self.spec, run_id="test_run"),
                             analyzer.analyze(self.spec, probes=self.capture))

    def test_reject_incomplete_capture(self):
        for change in ("missing", "short_keys", "all_tokens", "nan", "batch"):
            with self.subTest(change=change):
                capture = copy.deepcopy(self.capture)
                entry = capture["qk_cache"]["layer"]
                if change == "missing":
                    capture["qk_cache"] = {}
                elif change == "short_keys":
                    entry["k_all"] = entry["k_all"][:, :-1]
                elif change == "all_tokens":
                    entry["hookq_mode"] = "all_tokens"
                elif change == "nan":
                    entry["q"][0, 6] = float("nan")
                else:
                    entry["q"] = entry["q"].repeat(2, 1)
                with self.assertRaises(ValueError):
                    self.analyzer.analyze(self.spec, probes=capture)
        with self.assertRaises(ValueError):
            self.analyzer.analyze(self.spec)

    def test_reject_bad_spans(self):
        for span in ((1, 1), (-1, 2), (4, 6)):
            with self.subTest(span=span):
                self.spec["candidate_spans"][0] = span
                with self.assertRaisesRegex(ValueError, "spans"):
                    self.analyzer.analyze(self.spec, probes=self.capture)

    def test_final_block_and_multitoken_identifier(self):
        prompt = ("Example:\nCandidate Columns:\nexample@wrong\n\n"
                  "Now start:\nCandidate Columns:\n#table: t (\na@t\nlong name@t\n)\n")
        ids, spec = prepare_prompt(CharacterTokenizer(), prompt)
        self.assertEqual(spec["candidates"], ["a@t", "long name@t"])
        for name, (start, end) in zip(spec["candidates"], spec["candidate_spans"]):
            self.assertEqual("".join(map(chr, ids[start:end])), name)

    def test_reject_missing_empty_or_duplicate_candidates(self):
        for prompt in ("no candidates", "Candidate Columns:\n#table: t (\n)\n",
                       "Candidate Columns:\na@t\na@t\n"):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                prepare_prompt(CharacterTokenizer(), prompt)

    def test_ap_uses_full_ranking(self):
        ap, gold = evaluate_ranking(["a@t", "b@t", "c@t"], [1, 0, 2], ["T.`a`", "t.c"])
        self.assertAlmostEqual(ap, (1 / 2 + 2 / 3) / 2)
        self.assertEqual(gold, [True, False, True])
        with self.assertRaises(ValueError):
            evaluate_ranking(["a@t"], [0], ["t.missing"])
        with self.assertRaises(ValueError):
            evaluate_ranking(["a@t", "A@T"], [0, 1], ["t.a"])

    def test_temperature_normalizes_scores_and_preserves_ranking(self):
        unit = select_columns([9.0, 1.0], temperature=1.0)
        flat = select_columns([9.0, 1.0], temperature=2.0)
        self.assertAlmostEqual(unit["probabilities"][0], 0.9)
        self.assertAlmostEqual(sum(flat["probabilities"]), 1.0)
        self.assertLess(flat["probabilities"][0], unit["probabilities"][0])
        self.assertEqual(unit["ranking"], flat["ranking"])

    def test_top_p_boundary_and_ties(self):
        result = select_columns([1.0, 1.0, 1.0, 1.0], top_p=0.5)
        self.assertEqual(result["selected"], [0, 1])
        self.assertEqual(select_columns([1.0, 0.0], top_p=1.0)["selected"], [0, 1])
        self.assertEqual(select_columns([0.0, 0.0], top_p=0.01)["selected"], [0])
        self.assertTrue(all(math.isfinite(p) for p in
                            select_columns([9.0, 1.0], temperature=1e-300)["probabilities"]))

    def test_reject_invalid_selection_settings(self):
        for tau in (0, -1, float("nan"), float("inf")):
            with self.subTest(temperature=tau), self.assertRaises(ValueError):
                select_columns([1.0], temperature=tau)
        for top_p in (0, -1, 1.1, float("nan")):
            with self.subTest(top_p=top_p), self.assertRaises(ValueError):
                select_columns([1.0], top_p=top_p)
        for scores in ([], [-1.0], [float("nan")]):
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                select_columns(scores)

    def test_embedded_prompt_preserves_original_input(self):
        self.assertEqual(hashlib.sha256(INPUT_SEQ.encode()).hexdigest(),
                         "e31ea64cb8fa30384b88cfee1a3ec93dd16baa5a78662b1bf1085b30391998b2")


if __name__ == "__main__":
    unittest.main()
