"""Path resolution for standalone installs and in-repo development.

In the vLLM-Hook repo the ``live_highlighter`` package lives at
``notebooks/demo_token_highlighter/live_highlighter/``.

Precedence for **writes** (``download_dir``, hook artifacts under ``_v1_qk_peeks``):

1. ``cache_dir`` argument to :func:`resolve_runtime_paths`
2. ``LIVE_HIGHLIGHTER_CACHE`` environment variable
3. ``{repo_root}/cache`` when the vLLM-Hook repo is detected
4. ``~/.cache/live_highlighter`` (or ``$XDG_CACHE_HOME/live_highlighter``)

**Model snapshot lookup** scans the write cache, repo cache, and common Hugging Face hub
caches so existing weights are reused without a second download.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_CACHE = "LIVE_HIGHLIGHTER_CACHE"
ENV_CONFIG = "LIVE_HIGHLIGHTER_CONFIG"
IN_REPO_PACKAGE_REL = Path("notebooks") / "demo_token_highlighter" / "live_highlighter"


@dataclass(frozen=True)
class RuntimePaths:
  """Resolved filesystem layout for one model + highlighter run."""

  repo_root: Path | None
  cache_dir: Path
  config_path: Path
  model_hub: str
  model_path: str
  snapshot_cache: Path | None  # cache root that supplied ``model_path``, if local

  @property
  def hook_dir(self) -> Path:
    return self.cache_dir / "_v1_qk_peeks"

  def configure_hf(self) -> Path:
    """Set HF env vars (``setdefault``) to ``cache_dir`` before model/tokenizer load."""
    return configure_hf_cache(self.cache_dir)

  def hook_llm_kwargs(self, **overrides) -> dict:
    """Keyword args for :class:`vllm_hook_plugins.HookLLM` (override any field)."""
    base = {
      "model": self.model_path,
      "config_file": str(self.config_path),
      "download_dir": str(self.cache_dir),
      "worker_name": "token_highlighter",
      "analyzer_name": "token_highlighter",
      "enable_hook": True,
      "enforce_eager": True,
      "trust_remote_code": True,
    }
    base.update(overrides)
    return base

  def describe(self) -> str:
    lines = [
      f"model_hub       : {self.model_hub}",
      f"model_path      : {self.model_path}",
      f"cache_dir       : {self.cache_dir}",
      f"hook_dir        : {self.hook_dir}",
      f"config_path     : {self.config_path}",
    ]
    if self.repo_root:
      lines.append(f"repo_root       : {self.repo_root}")
    if self.snapshot_cache:
      lines.append(f"snapshot_cache  : {self.snapshot_cache}")
    return "\n".join(lines)


def find_repo_root(cwd: Path | str | None = None) -> Path | None:
  """Return vLLM-Hook repo root if ``vllm_hook_plugins`` and ``model_configs`` exist."""
  start = Path(cwd or Path.cwd()).resolve()
  for root in (start, *start.parents):
    if (root / "vllm_hook_plugins").is_dir() and (root / "model_configs").is_dir():
      return root
  return None


def find_live_highlighter_dir(
  *,
  cwd: Path | str | None = None,
  repo_root: Path | None = None,
) -> Path | None:
  """Return the in-repo ``live_highlighter`` package directory when present."""
  root = repo_root if repo_root is not None else find_repo_root(cwd)
  if root is not None:
    candidate = (root / IN_REPO_PACKAGE_REL).resolve()
    if (candidate / "setup.py").is_file():
      return candidate

  start = Path(cwd or Path.cwd()).resolve()
  for base in (start, *start.parents):
    if base.name == "live_highlighter" and (base / "setup.py").is_file():
      return base.resolve()
  return None


def _default_user_cache() -> Path:
  xdg = os.environ.get("XDG_CACHE_HOME")
  base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
  return (base / "live_highlighter").resolve()


def _known_hub_caches() -> list[Path]:
  """Existing Hugging Face cache directories (deduped, newest candidates first)."""
  seen: set[Path] = set()
  out: list[Path] = []
  for key in ("HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
    val = os.environ.get(key)
    if not val:
      continue
    p = Path(val).expanduser().resolve()
    if p.is_dir() and p not in seen:
      seen.add(p)
      out.append(p)
  hf_hub = (Path.home() / ".cache" / "huggingface" / "hub").resolve()
  if hf_hub.is_dir() and hf_hub not in seen:
    out.append(hf_hub)
  return out


def _snapshot_in_cache(model_hub: str, cache: Path | str) -> str | None:
  if Path(model_hub).is_dir():
    return str(Path(model_hub).resolve())
  snaps = Path(cache) / f"models--{model_hub.replace('/', '--')}" / "snapshots"
  if not snaps.is_dir():
    return None
  candidates = sorted(snaps.iterdir())
  if not candidates:
    return None
  return str(candidates[-1].resolve())


def find_model_path(
  model_hub: str,
  *,
  write_cache: Path,
  repo_root: Path | None = None,
) -> tuple[str, Path | None]:
  """Return ``(model_path, snapshot_cache)`` — local snapshot if found, else hub id."""
  if Path(model_hub).is_dir():
    return str(Path(model_hub).resolve()), None

  search_roots: list[Path] = []
  for root in (write_cache, repo_root / "cache" if repo_root else None, *_known_hub_caches()):
    if root is None:
      continue
    root = Path(root).resolve()
    if root not in search_roots:
      search_roots.append(root)

  for root in search_roots:
    snap = _snapshot_in_cache(model_hub, root)
    if snap:
      return snap, root
  return model_hub, None


def _resolve_cache_dir(
  *,
  explicit: Path | str | None,
  repo_root: Path | None,
) -> Path:
  if explicit is not None:
    p = Path(explicit).expanduser().resolve()
  elif os.environ.get(ENV_CACHE):
    p = Path(os.environ[ENV_CACHE]).expanduser().resolve()
  elif repo_root is not None:
    p = (repo_root / "cache").resolve()
  else:
    p = _default_user_cache()
  p.mkdir(parents=True, exist_ok=True)
  return p


def _resolve_config_path(
  model_hub: str,
  *,
  explicit: Path | str | None,
  repo_root: Path | None,
) -> Path:
  if explicit is not None:
    p = Path(explicit).expanduser().resolve()
    if not p.is_file():
      raise FileNotFoundError(f"Highlighter config not found: {p}")
    return p

  if os.environ.get(ENV_CONFIG):
    p = Path(os.environ[ENV_CONFIG]).expanduser().resolve()
    if not p.is_file():
      raise FileNotFoundError(f"{ENV_CONFIG} points to missing file: {p}")
    return p

  short = model_hub.split("/")[-1]
  if repo_root is not None:
    p = repo_root / "model_configs" / "token_highlighter" / f"{short}.json"
    if p.is_file():
      return p.resolve()

  raise FileNotFoundError(
    f"No highlighter config for {model_hub}. "
    f"Set {ENV_CONFIG} or run from the vLLM-Hook repo "
    f"(expected model_configs/token_highlighter/{short}.json)."
  )


def resolve_runtime_paths(
  model_hub: str,
  *,
  cache_dir: Path | str | None = None,
  config_path: Path | str | None = None,
  cwd: Path | None = None,
) -> RuntimePaths:
  """Resolve cache, config, and model snapshot paths for HookLLM + visualizer."""
  repo_root = find_repo_root(cwd)
  write_cache = _resolve_cache_dir(explicit=cache_dir, repo_root=repo_root)
  model_path, snapshot_cache = find_model_path(
    model_hub, write_cache=write_cache, repo_root=repo_root
  )
  cfg = _resolve_config_path(model_hub, explicit=config_path, repo_root=repo_root)
  return RuntimePaths(
    repo_root=repo_root,
    cache_dir=write_cache,
    config_path=cfg,
    model_hub=model_hub,
    model_path=model_path,
    snapshot_cache=snapshot_cache,
  )


def configure_hf_cache(cache: Path | str) -> Path:
    """Point Hugging Face env vars at ``cache`` (``setdefault`` only)."""
    cache = Path(cache).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache))
    os.environ.setdefault("HF_HUB_CACHE", str(cache))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache))
    return cache
