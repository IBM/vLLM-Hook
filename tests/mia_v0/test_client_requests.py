"""What MiaClient actually puts on the wire.

The client's whole job is to turn a call into one request whose `vllm_xargs` the plugin will
read back correctly, so these tests stub the OpenAI transport and assert on the request --
no server, no GPU. The cases that matter are the ones a wrong answer would hide: a dict that
the plugin does not JSON-decode would arrive as a string and be silently ignored, and a
token-exact prompt that went through the chat endpoint would be re-tokenized.
"""
from __future__ import annotations

import json
import types

import pytest

pytest.importorskip("vllm")

from mia.client import MiaClient                                        # noqa: E402

MODEL = "Qwen/Qwen2-1.5B-Instruct"


class _Recorder:
    """Stands in for `openai.OpenAI`, recording what each endpoint was called with."""

    def __init__(self):
        self.calls = []
        outer = self

        def _create(**kwargs):
            outer.calls.append(kwargs)
            return types.SimpleNamespace(probes=None, model_dump_json=lambda: "{}")

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))
        self.completions = types.SimpleNamespace(create=_create)

    @property
    def last(self):
        return self.calls[-1]

    def last_xargs(self):
        body = self.last["extra_body"]
        return None if body is None else body["vllm_xargs"]


@pytest.fixture
def client(tmp_path):
    cfg = tmp_path / "hs.json"
    cfg.write_text(json.dumps({
        "model_info": {"name": MODEL},
        "hidden_states": {"layers": [1, 2], "mode": "last_token"},
    }))
    c = MiaClient(base_url="http://127.0.0.1:1/v1", analyzer_name="hidden_states",
                  config_file=str(cfg), hook_dir=str(tmp_path / "artifacts"))
    c._openai = _Recorder()
    return c


def test_chat_request_carries_the_capture_xargs(client):
    client.generate(messages=[{"role": "user", "content": "hi"}], model=MODEL,
                    save_to_disk=True, run_id="r1")
    xargs = client._openai.last_xargs()
    assert xargs["output_hidden_states"] == json.dumps([1, 2])
    assert xargs["run_id"] == "r1"
    assert xargs["save_to_disk"] is True
    assert xargs["hook_dir"].endswith("artifacts")


def test_scalar_extra_xargs_passes_through(client):
    """`hooks_on` is the one the long-decode demos need; it must arrive verbatim."""
    client.generate(messages=[], model=MODEL, extra_xargs={"hooks_on": "both"})
    assert client._openai.last_xargs()["hooks_on"] == "both"


def test_a_dict_under_an_undecoded_key_is_refused(client):
    """The plugin JSON-decodes three keys only; anything else would reach it as a string."""
    with pytest.raises(ValueError, match="only JSON-decodes"):
        client.generate(messages=[], model=MODEL, extra_xargs={"whatever": {"a": 1}})


def test_a_dict_under_a_decoded_key_is_json_encoded(client):
    client.generate(messages=[], model=MODEL, extra_xargs={"output_qk": {"0": [1, 2]}})
    assert json.loads(client._openai.last_xargs()["output_qk"]) == {"0": [1, 2]}


def test_steer_is_json_encoded_under_its_own_key(client):
    steer = {"method": "add_vector", "coefficient": 10}
    client.generate(messages=[], model=MODEL, steer=steer)
    assert json.loads(client._openai.last_xargs()["steer"]) == steer


def test_capture_false_sends_no_xargs_at_all(client):
    """The serve-path equivalent of offline `use_hook=False`: arm nothing."""
    client.generate(messages=[], model=MODEL, capture=False)
    assert client._openai.last_xargs() is None


def test_capture_false_still_carries_a_steer_config(client):
    """Steering produces no artifact, so it is independent of capture."""
    client.generate(messages=[], model=MODEL, capture=False, steer={"coefficient": 1})
    xargs = client._openai.last_xargs()
    assert set(xargs) == {"steer"}


def test_generate_tokens_uses_the_completions_endpoint_verbatim(client):
    """Token ids must not be re-tokenized, which is why this is not the chat endpoint."""
    ids = [1, 2, 3, 4]
    client.generate_tokens(ids, model=MODEL, save_to_disk=True)
    assert client._openai.last["prompt"] == ids
    assert client._openai.last_xargs()["save_to_disk"] is True


def test_generate_tokens_accepts_a_batch_under_one_run_id(client):
    batch = [[1, 2], [3, 4, 5]]
    client.generate_tokens(batch, model=MODEL, run_id="shared", save_to_disk=True)
    assert client._openai.last["prompt"] == batch
    assert client._openai.last_xargs()["run_id"] == "shared"


def test_generate_text_skips_the_chat_template(client):
    client.generate_text("already templated", model=MODEL)
    assert client._openai.last["prompt"] == "already templated"


def test_analyze_accepts_probes_without_any_request(client):
    """Mirrors the offline `analyze(probes=...)`; must not require a prior generate()."""
    seen = {}
    client.analyzer = types.SimpleNamespace(
        analyze=lambda spec, probes=None: seen.update(spec=spec, probes=probes) or "ok")
    assert client.analyze(analyzer_spec={"reduce": "none"}, probes={"hs_cache": {}}) == "ok"
    assert seen["probes"] == {"hs_cache": {}}


def test_tokenizer_without_a_model_id_says_what_to_do(client):
    with pytest.raises(RuntimeError, match="tokenizer_for"):
        client.tokenizer
