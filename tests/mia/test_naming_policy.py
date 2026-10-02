"""The rename is policy, not vibes: nothing in the tree may carry the old brand.

Referring to the DEPENDENCY vLLM (import vllm, VLLM_USE_V2_MODEL_RUNNER, "runs on
vLLM 0.29") is allowed — only MIA's own names are policed.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# (pattern, human explanation)
BANNED_PATTERNS = [
    # Separator-agnostic: the old brand was written `vLLM-Hook`, `vllm_hook`, `vLLM.hook`
    # and `VLLM HOOK` in different places, and a pattern pinned to `[_-]` passed 8 of them.
    (re.compile(r"vllm[\s._-]{0,2}hook", re.I), "old package/brand name -> mia"),
    (re.compile(r"VLLM_HOOK_"), "old env-var prefix -> MIA_"),
    (re.compile(r"HookLLM|HookClient|HookHost"), "old public class names"),
    # Separator-agnostic for the same reason as the brand above: pinned to the underscore, this
    # pattern passed the separator-less `hookplugin`, and 15 user-visible print prefixes kept the
    # old brand while the whole suite went green.
    (re.compile(r"hook[\s._-]{0,2}plugin", re.I), "old plugin module name -> _plugin"),
    # 'ring' as a TOKEN only: hits capture_ring / RING_DIR / GpuCaptureRing / OffLoopQKRingDrain,
    # spares steering_vectors, STEERING, bring, during, String. Validated before this plan ran.
    (re.compile(r"(?<![A-Za-z])ring(?![a-z])|(?<![A-Z])RING(?![a-z])|Ring(?![a-z])"),
     "'ring' is retired -> aperture"),
]

# Paths that legitimately keep old strings.
EXEMPT_DIRS = {".git", "__pycache__", ".pytest_cache", "docs/superpowers", "cache", ".superpowers"}
EXEMPT_FILES = {"tests/mia/test_naming_policy.py"}  # spells the banned names in order to police them
SCANNED_SUFFIXES = {".py", ".sh", ".md", ".txt", ".cfg", ".toml", ".json", ".yaml", ".yml", ".ipynb"}

# Legitimate survivors, each with a reason. Narrow by construction: a hit is excused only when it
# falls INSIDE one of these spans on its own line, never merely because the line mentions one.
#  * the checkout directories are deliberately not renamed (§0.2 non-goal), so every script that
#    points at them carries "vLLM-Hook" inside a PATH;
#  * the profiling repo keeps its own name;
#  * `vllm_hook_env` is a real pre-existing conda env (the 0.21 validation arm) — renaming the text
#    without renaming the env would send a reader to an env that does not exist.
#  * the PREPRINT's published title is a citation. Renaming it would misquote a paper that
#    exists under that title at that DOI; a citation records what was published, not what the
#    project is called now. Both spellings appear (markdown-bold in the badge, plain in the
#    BibTeX), and the span is pinned to the full title so it excuses the citation and nothing
#    else -- "vLLM Hook" anywhere other than that exact sentence is still a violation.
ALLOWED_CONTEXTS = [
    re.compile(r"[~\w./-]*/vLLM-Hook(?:-Profiling)?\b[\w./-]*", re.I),
    re.compile(r"\bvLLM-Hook-Profiling\b", re.I),
    re.compile(r"\bvllm_hook_env\b"),
    re.compile(r"\bvllm_hook_profiling\b"),  # the profiler's permanent package name
    re.compile(r"\*{0,2}vLLM Hook\*{0,2} v0: A Plug-in for Programming Model Internals on vLLM"),
]


# UPSTREAM DOCUMENTS -- theirs, preserved on purpose.
#
# This fork proposes the rename in its pull request to IBM/vLLM_Hook; it does not assert it
# inside IBM's own prose. So these files keep upstream's wording, and the old brand in them is
# THEIRS rather than a leak of ours. They are excused for CONTENT only -- never for their path,
# and never for any other file -- and they stay IN the scan, so
# `test_the_scan_actually_reaches_files` still proves the walk reaches README.md.
#
# Self-expiring, like the quarantine below: if one of these is ever rebranded, it stops
# violating and `test_every_upstream_document_is_still_violating` fails on the stale entry.
UPSTREAM_DOCS: dict[str, str] = {
    "README.md":
        "IBM's project README -- title, tagline, ICML entry, benchmark section and the closing "
        "'started by IBM Research' line are upstream's voice; only statements naming a real "
        "symbol (`mia`, `MiaLLM`, `MIA_*`) were updated.",
    "docs/configs.md":
        "IBM's configuration guide -- the project name in prose is upstream's; every `MIA_*` "
        "env var in it is a real symbol and is still policed by the other patterns.",
    ".github/PULL_REQUEST_TEMPLATE.md":
        "IBM's PR template, restored byte-for-byte. It names the pre-rename core files "
        "(`hook_llm.py`, `_hook_plugin.py`, `hook_client.py`) and their technical report; a "
        "contributor's template is not ours to edit.",
}


# QUARANTINE -- known, enumerated, and self-expiring.
#
# `mia/` is frozen while a GPU job runs against the live tree, so this one violation cannot be
# fixed in the same change that widened the pattern which found it. Suppressing it silently
# (an EXEMPT_FILES entry) would make it permanent by accident, and leaving the gate red would
# make the whole suite red for a file this change is not allowed to touch. So it is listed,
# with its reason, and TWO tests hold it in place: one asserts the list has not grown, the
# other asserts each entry is STILL violating -- so when `mia/` is fixed, the stale entry
# fails and has to be removed. A quarantine that outlives its cause is just an exemption.
QUARANTINE: dict[str, str] = {
    # EMPTY, and it must stay that way. The one entry this mechanism was built for
    # (mia/utils/spotlight/utils.py, frozen behind a running GPU job) was fixed the moment
    # mia/ unfroze, and `test_every_quarantined_file_is_still_violating` duly went red on the
    # stale entry -- which is the mechanism working, so the entry came out in the same commit
    # as the fix. The dict and its two tests stay: an empty quarantine with a live
    # "must be empty" assertion is a guard, whereas deleting the machinery would let the next
    # quarantine be added with none.
}


def _excused(line: str, start: int, end: int) -> bool:
    return any(m.start() <= start and end <= m.end()
               for ctx in ALLOWED_CONTEXTS for m in ctx.finditer(line))


def _iter_files():
    for path in REPO.rglob("*"):
        if not path.is_file() or path.suffix not in SCANNED_SUFFIXES:
            continue
        rel = path.relative_to(REPO).as_posix()
        if any(rel == f for f in EXEMPT_FILES):
            continue
        if any(rel == d or rel.startswith(d + "/") for d in EXEMPT_DIRS):
            continue
        # *.egg-info/ is generated build output that regenerates on every
        # `pip install -e .`; exempt any such directory by suffix (not a
        # single hard-coded package name) so the gate's count doesn't
        # depend on local build state.
        if any(part.endswith(".egg-info") for part in rel.split("/")[:-1]):
            continue
        yield path, rel


def scan_tree(root: Path = REPO, *, include_quarantined: bool = False):
    """Return [(relpath, lineno, why)] for every policy violation, contents and names.

    `include_quarantined=True` returns the raw truth, which is what the quarantine tests
    below check themselves against.
    """
    hits = []
    for path, rel in _iter_files():
        for pattern, why in BANNED_PATTERNS:
            if pattern.search(rel):
                hits.append((rel, 0, f"path: {why}"))
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for pattern, why in BANNED_PATTERNS:
                for m in pattern.finditer(line):
                    if _excused(line, m.start(), m.end()):
                        continue
                    hits.append((rel, lineno, why))
    if not include_quarantined:
        hits = [h for h in hits if h[0] not in QUARANTINE and h[0] not in UPSTREAM_DOCS]
    return hits


def test_the_scan_actually_reaches_files():
    """Guard the guard: a scan that reaches nothing reports no violations and passes.

    Every exclusion in `_iter_files` -- the suffix set, EXEMPT_DIRS, EXEMPT_FILES, the
    `.egg-info` rule -- is a way for this gate to quietly stop looking, and the gate below
    asserts an EMPTY list, so it gets greener the less it scans. The same guard exists in
    `tests/test_parity_band_index.py` for the same reason; it was missing here.
    """
    scanned = [rel for _path, rel in _iter_files()]
    assert len(scanned) > 100, (
        f"the naming scan reached only {len(scanned)} file(s): {scanned[:10]}. It should "
        f"cover the whole tree; an exclusion has swallowed it, and an empty scan passes the "
        f"policy gate without checking anything.")
    # ...and it must reach the places the old brand actually lived.
    for expected in ("README.md", "tests/mia/parity/capture_workload.py"):
        assert expected in scanned, f"{expected} is not being scanned"
    assert any(rel.startswith("notebooks/") for rel in scanned), "notebooks are not scanned"


def test_the_policy_pattern_catches_every_old_spelling():
    """The pattern is the gate. Pin the spellings that were missed, so a future narrowing
    fails here rather than by letting the brand back in.
    """
    brand = BANNED_PATTERNS[0][0]
    for spelling in ("vLLM-Hook", "vllm_hook", "vLLM.hook", "VLLM HOOK", "vllmhook"):
        assert brand.search(spelling), f"{spelling!r} is not caught by the brand pattern"
    for allowed in ("vllm", "hook", "forward_hook", "register_forward_hook"):
        assert not brand.search(allowed), f"{allowed!r} must NOT be caught"


def test_the_plugin_module_pattern_catches_the_separatorless_spelling():
    """`hookplugin` is the spelling that got through: the pattern required the underscore.

    It let 15 user-visible print prefixes keep the old brand while the whole suite passed, which
    is the exact failure mode the brand pattern above was already widened for.
    """
    plugin = dict((why, pat) for pat, why in BANNED_PATTERNS)["old plugin module name -> _plugin"]
    for spelling in ("hook_plugin", "hookplugin", "hook plugin", "hook-plugin", "HookPlugin"):
        assert plugin.search(spelling), f"{spelling!r} is not caught by the plugin pattern"
    for allowed in ("plugin", "register_plugins", "PluginRegistry", "general_plugins"):
        assert not plugin.search(allowed), f"{allowed!r} must NOT be caught"


def test_the_citation_exemption_is_narrow():
    """The preprint title is excused; the brand elsewhere on the same line is not."""
    citation = "[**vLLM Hook** v0: A Plug-in for Programming Model Internals on vLLM](x)"
    hit = BANNED_PATTERNS[0][0].search(citation)
    assert hit and _excused(citation, hit.start(), hit.end())
    other = "vLLM Hook is the old name of this project"
    hit = BANNED_PATTERNS[0][0].search(other)
    assert hit and not _excused(other, hit.start(), hit.end())


def test_the_quarantine_is_empty():
    """The strongest form of "has not grown": nothing is exempt at all.

    Every entry here is a violation the gate is NOT enforcing, so re-opening the quarantine
    has to be a deliberate, reviewed act that fails this test first. It was emptied when
    mia/ unfroze and the one quarantined file was fixed.
    """
    assert QUARANTINE == {}, (
        f"the naming-policy quarantine now holds {sorted(QUARANTINE)}. Every entry is a "
        f"violation the gate is NOT enforcing; fix the file instead. If a file genuinely "
        f"cannot be touched right now, say why here and in the entry's reason string.")


def test_every_quarantined_file_is_still_violating():
    """...and the check that makes the quarantine self-expiring.

    A suppression that outlives the problem it suppressed is indistinguishable from an
    exemption, and it silently re-opens the hole for the next occurrence in that file.

    Vacuous while QUARANTINE is empty -- which is the desired state, and which
    `test_the_quarantine_is_empty` above asserts directly. This one is the check that arms
    itself the moment anyone adds an entry, so it is kept rather than deleted with the entry
    it expired.
    """
    raw = {rel for rel, _lineno, _why in scan_tree(include_quarantined=True)}
    for rel, reason in sorted(QUARANTINE.items()):
        assert rel in raw, (
            f"{rel} is quarantined but no longer violates the naming policy -- remove it "
            f"from QUARANTINE. (Reason it was added: {reason})")


def test_upstream_documents_are_exactly_the_three_we_preserve():
    """Widening this set is how our own brand leak would get excused, so it is pinned."""
    assert set(UPSTREAM_DOCS) == {
        "README.md", "docs/configs.md", ".github/PULL_REQUEST_TEMPLATE.md"}, (
        f"UPSTREAM_DOCS now holds {sorted(UPSTREAM_DOCS)}. Only documents OWNED BY UPSTREAM and "
        f"deliberately left un-rebranded belong here; anything else is our own leak and must be "
        f"fixed in the file instead.")
    assert all(reason.strip() for reason in UPSTREAM_DOCS.values()), "every entry needs a reason"


def test_every_upstream_document_is_still_violating():
    """Self-expiring: a rebranded upstream doc no longer needs the excuse, so the entry must go.

    Without this, the set would silently become a permanent licence for the old brand to live in
    files that no longer carry upstream's wording at all.
    """
    raw = {rel for rel, _lineno, _why in scan_tree(include_quarantined=True)}
    for rel, reason in sorted(UPSTREAM_DOCS.items()):
        assert rel in raw, (
            f"{rel} is listed as an un-rebranded upstream document but carries no legacy name "
            f"any more -- remove it from UPSTREAM_DOCS. (Reason it was added: {reason})")


def test_no_legacy_names_anywhere():
    hits = scan_tree()
    report = "\n".join(f"  {rel}:{lineno}  {why}" for rel, lineno, why in hits[:60])
    assert not hits, (f"{len(hits)} naming-policy violations "
                      f"(excluding {len(QUARANTINE)} quarantined):\n{report}")
