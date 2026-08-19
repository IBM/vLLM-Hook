# tests/core/test_canonical.py
import hashlib

from vllm_hook_plugins.core.canonical import canonical_bytes, request_salt, spec_hash


def test_canonical_bytes_sorts_keys_and_strips_whitespace():
    assert canonical_bytes({"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}'


def test_canonical_bytes_is_key_order_independent():
    assert canonical_bytes({"x": 1, "y": {"b": 2, "a": 3}}) == canonical_bytes(
        {"y": {"a": 3, "b": 2}, "x": 1}
    )


def test_spec_hash_fixture():
    obj = {"ops": []}
    expected = hashlib.sha256(b'{"ops":[]}').hexdigest()
    assert spec_hash(obj) == f"sha256:{expected}"


def test_request_salt_is_artifact_order_independent():
    spec = {"ops": []}
    a = "sha256:" + "aa" * 32
    b = "sha256:" + "bb" * 32
    assert request_salt(spec, [a, b]) == request_salt(spec, [b, a])


def test_request_salt_fixture():
    spec = {"ops": []}
    artifact = "sha256:" + "aa" * 32
    h = hashlib.sha256(b'{"ops":[]}')
    h.update(artifact.encode("utf-8"))
    assert request_salt(spec, [artifact]) == h.hexdigest()


def test_request_salt_varies_with_spec_and_artifacts():
    a = "sha256:" + "aa" * 32
    b = "sha256:" + "bb" * 32
    base = request_salt({"ops": []}, [a])
    assert request_salt({"ops": [1]}, [a]) != base
    assert request_salt({"ops": []}, [b]) != base
    assert len(base) == 64
    assert all(c in "0123456789abcdef" for c in base)
