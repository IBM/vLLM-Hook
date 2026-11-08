"""Locate the first tool call in a generated action, for decision-time probes.

A decision-time probe reads the decode states of the action. Models differ in where the call
sits. Some write it first, some write a sentence of text first, and some put the arguments
before the tool name, so a fixed window of the first generated tokens can miss it.
``tool_call_window`` returns the token window of the first call, which a probe with
``"position": "decode_window"`` then reads (see ``mia.analyzers.two_stage_probe_analyzer``).

A call is the first top-level JSON value in the text that is an object, or a list holding one,
with a ``"name"`` and an ``"arguments"`` or ``"parameters"`` key. A list of several calls gives one
window over the whole list, and a call nested inside another JSON object is not found. The call
starts at a ``<tool_call>`` or ``<|tool_call|>`` tag written directly before that JSON. A tag with
no parsable JSON after it still marks a call that runs to the end of the text.
"""
from __future__ import annotations

import json
from typing import Optional, Sequence, Tuple

TOOL_CALL_TAGS = ("<tool_call>", "<|tool_call|>")


def find_tool_call(text: str) -> Optional[Tuple[int, int]]:
    """Character span ``(start, end)`` of the first tool call in ``text``, or None."""
    dec, i = json.JSONDecoder(), 0
    while i < len(text):
        j = min([k for k in (text.find("{", i), text.find("[", i)) if k >= 0], default=-1)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text[j:])
        except json.JSONDecodeError:
            i = j + 1
            continue
        items = obj if isinstance(obj, list) else [obj]
        if any(isinstance(d, dict) and "name" in d and ("arguments" in d or "parameters" in d) for d in items):
            start = j
            for tag in TOOL_CALL_TAGS:
                k = text.rfind(tag, 0, j)
                if k >= 0 and not text[k + len(tag):j].strip():
                    start = min(start, k)
            return start, j + end
        i = j + max(end, 1)
    tags = [k for k in (text.find(t) for t in TOOL_CALL_TAGS) if k >= 0]
    return (min(tags), len(text)) if tags else None


def _first_token_ending_after(tokenizer, token_ids: Sequence[int], char: int) -> int:
    """Smallest t such that the decoded text of token_ids[:t + 1] is longer than ``char``."""
    lo, hi = 0, len(token_ids) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(token_ids[:mid + 1], skip_special_tokens=False)) > char:
            hi = mid
        else:
            lo = mid + 1
    return lo


def tool_call_window(tokenizer, token_ids: Sequence[int], max_tokens: int = 64) -> Optional[Tuple[int, int]]:
    """Window ``(start, end)`` of generated-token indices covering the first tool call.

    ``start`` is the token holding the call's first character, ``end`` is one past the token
    holding its last character, and the window is cut to ``max_tokens``. The last generated token
    is never fed back to the model, so it has no decode state, and the window is also cut to end
    before it. Returns None when the action holds no call, or when the call starts at the last
    token. Token positions come from decoding prefixes of ``token_ids`` (a binary search, about
    2 log2(n) decodes), so they index the generated tokens themselves.
    """
    token_ids = list(token_ids)
    if not token_ids:
        return None
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    span = find_tool_call(text)
    if span is None:
        return None
    start = _first_token_ending_after(tokenizer, token_ids, span[0])
    last = _first_token_ending_after(tokenizer, token_ids, span[1] - 1)
    end = min(last + 1, start + max_tokens, len(token_ids) - 1)
    return (start, end) if start < end else None
