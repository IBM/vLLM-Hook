"""Server-mode helpers for the demos' commented `vllm serve` blocks and `demo_actsteer_serve.py`.

One server serves one worker kind, chosen at launch with ``MIA_WORKER``. :func:`require_server`
prints the exact `vllm serve` command and exits when nothing is listening.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence

#: Worker kinds `MIA_WORKER` accepts (exact; MIA refuses anything else).
HS, QK, STEER = "hidden_states", "qk", "steer"

#: vLLM's GPU share on a capture server; the rest holds the capture aperture (4 GiB by default).
CAPTURE_GPU_MEMORY_UTILIZATION = 0.8


def base_url() -> str:
    """The server's OpenAI-compatible endpoint; override with ``MIA_DEMO_BASE_URL``."""
    return os.environ.get("MIA_DEMO_BASE_URL", "http://localhost:8770/v1")


def serve_command(model: str, worker: str, *, graph: bool = True,
                  max_model_len: int = 2048, tp: int = 1,
                  extra_args: Sequence[str] = ()) -> str:
    """The `vllm serve` command a demo needs, ready to paste.

    `graph=False` adds `--enforce-eager`; `tp` > 1 adds `--tensor-parallel-size`; `extra_args`
    are flags the demo needs (e.g. `--no-enable-prefix-caching`).
    """
    env = ["VLLM_WORKER_MULTIPROC_METHOD=spawn", f"MIA_WORKER={worker}"]
    args = [f"--max-model-len {max_model_len}", f"--port {_port()}"]
    if worker in (HS, QK):
        args.append(f"--gpu-memory-utilization {CAPTURE_GPU_MEMORY_UTILIZATION}")
    if int(tp) > 1:
        args.append(f"--tensor-parallel-size {int(tp)}")
    args.extend(extra_args)
    if not graph:
        # Opt-out of MIA's CUDA-graph default; eager capture is bit-exact.
        args.append("--enforce-eager")
    return f"{' '.join(env)} \\\n    vllm serve {model} \\\n    {' '.join(args)}"


def require_server(model: str, worker: str, *, graph: bool = True,
                   max_model_len: int = 2048, tp: int = 1,
                   extra_args: Sequence[str] = (), model_env: bool = False) -> str:
    """Return the base URL, or explain how to start the server and exit.

    `model_env=True` for a demo that reads ``MIA_DEMO_MODEL``: a model mismatch then names it.
    """
    url = base_url()
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as r:
            served = {m.get("id") for m in json.loads(r.read()).get("data", [])}
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[mia] no server at {url} ({e}).\n\nStart one in another terminal:\n\n"
              f"{serve_command(model, worker, graph=graph, max_model_len=max_model_len, tp=tp, extra_args=extra_args)}\n",
              file=sys.stderr)
        raise SystemExit(1)

    if served and model not in served:
        hint = " or set MIA_DEMO_MODEL to one it serves" if model_env else ""
        print(f"[mia] the server at {url} serves {sorted(served)}, not {model!r}. Restart it "
              f"for this model{hint}.", file=sys.stderr)
        raise SystemExit(1)

    print(f"[mia] server at {url}, model {model}, MIA_WORKER={worker}"
          f"{'' if graph else ' (eager)'}")
    return url


def print_evidence(elapsed_s: float, n_tokens: int, label: str = "") -> None:
    """Client-side timing. The capture counters live in the server process, not here."""
    tag = f"[evidence:{label}]" if label else "[evidence]"
    per_tok = (elapsed_s * 1000 / n_tokens) if n_tokens else float("nan")
    print(f"{tag} {elapsed_s * 1000:.1f} ms for {n_tokens} tokens "
          f"({per_tok:.2f} ms/token, measured at the client)")
    print(f"{tag} capture counters are server-side: start the server with MIA_PROFILE=1; at exit "
          f"it writes them to $MIA_PROFILE_DIR (default /tmp/mia_profile).")


def chat(prompt: str) -> list:
    """A one-turn chat message list, which is what the server endpoint takes."""
    return [{"role": "user", "content": prompt}]


def completion_text(response) -> str:
    """The text of a chat completion's first choice."""
    return response.choices[0].message.content or ""


def completion_tokens(response) -> int:
    """How many tokens the server generated for this response (0 when not reported)."""
    usage = getattr(response, "usage", None)
    return int(getattr(usage, "completion_tokens", 0) or 0)


def _port() -> str:
    """The port in ``base_url()``, else the default 8770."""
    url = base_url()
    tail = url.rstrip("/").rsplit(":", 1)[-1]
    return tail.split("/")[0] if tail[:1].isdigit() else "8770"


__all__ = ["HS", "QK", "STEER", "CAPTURE_GPU_MEMORY_UTILIZATION", "base_url", "serve_command",
           "require_server", "print_evidence", "chat", "completion_text", "completion_tokens"]
