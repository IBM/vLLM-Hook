# tests/core/test_fingerprints.py
import hashlib

from vllm_hook_plugins.core.fingerprints import (
    chat_template_fingerprint,
    config_fingerprint,
    tokenizer_fingerprint,
)


def test_chat_template_recipe():
    assert chat_template_fingerprint("hello") == (
        "sha256:" + hashlib.sha256(b"hello").hexdigest()
    )
    # None and '' hash identically by design
    assert chat_template_fingerprint(None) == chat_template_fingerprint("")


def test_tokenizer_recipe_fixture(tmp_path):
    vocab = tmp_path / "vocab.json"
    merges = tmp_path / "merges.txt"
    vocab.write_bytes(b'{"a": 0}')
    merges.write_bytes(b"a b")

    digest = hashlib.sha256()
    for path in (merges, vocab):  # sorted by basename: merges.txt < vocab.json
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("utf-8"))
    expected = "sha256:" + digest.hexdigest()

    assert tokenizer_fingerprint([str(vocab), str(merges)]) == expected
    # input order must not matter
    assert tokenizer_fingerprint([str(merges), str(vocab)]) == expected


def test_tokenizer_fingerprint_tracks_content(tmp_path):
    f = tmp_path / "tokenizer.json"
    f.write_bytes(b"one")
    first = tokenizer_fingerprint([str(f)])
    f.write_bytes(b"two")
    assert tokenizer_fingerprint([str(f)]) != first


def test_config_recipe_ignores_volatile_fields():
    base = {"hidden_size": 64, "num_hidden_layers": 2}
    with_volatile = dict(base, _name_or_path="/tmp/x", transformers_version="4.99")
    assert config_fingerprint(base) == config_fingerprint(with_volatile)
    assert config_fingerprint(dict(base, hidden_size=128)) != config_fingerprint(base)


def test_config_recipe_is_key_order_independent():
    a = {"hidden_size": 64, "num_hidden_layers": 2}
    b = {"num_hidden_layers": 2, "hidden_size": 64}
    assert config_fingerprint(a) == config_fingerprint(b)
