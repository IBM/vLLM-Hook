"""Shared plumbing for MIA's server-mode demos.

MIA installs into the **server**: the worker that captures or steers is `vllm serve`'s own
worker, so every demo here talks to a server you start yourself. The plugin is selected at
launch with ``MIA_WORKER``, which is why one server serves one worker kind at a time.

Each demo calls :func:`require_server` first. If nothing is listening it prints the exact
command for that demo and exits, rather than failing later with a connection error.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

#: Worker kinds `MIA_WORKER` accepts (exact; MIA refuses anything else).
HS, QK, STEER = "hidden_states", "qk", "steer"


def base_url() -> str:
    """The server's OpenAI-compatible endpoint; override with ``MIA_DEMO_BASE_URL``."""
    return os.environ.get("MIA_DEMO_BASE_URL", "http://localhost:8770/v1")


def serve_command(model: str, worker: str, *, graph: bool = True,
                  max_model_len: int = 2048, tp: int = 1) -> str:
    """The `vllm serve` invocation a demo needs, ready to paste.

    `tp` > 1 adds `--tensor-parallel-size`. Capture shards across the ranks -- hidden states
    by layer, Q/K by head -- and each rank writes its own `tp_rank_<r>/`. Note that
    MIA_APERTURE_GPU_BYTES is a PER-RANK budget, so TP x N claims N times that much GPU.
    """
    env = ["VLLM_WORKER_MULTIPROC_METHOD=spawn", f"MIA_WORKER={worker}"]
    args = [f"--max-model-len {max_model_len}", f"--port {_port()}"]
    if int(tp) > 1:
        args.append(f"--tensor-parallel-size {int(tp)}")
    if graph:
        # MIA's point is capture that survives CUDA graphs, so this is the default. The mode
        # must be named explicitly: MIA accepts NONE or FULL, and vLLM would otherwise pick a
        # mode MIA refuses.
        env.append("MIA_ALLOW_CUDAGRAPH=1")
        args.append("""--compilation-config '{"cudagraph_mode": "FULL"}'""")
    else:
        # Opt-out. Eager capture is bit-exact, which graph-mode capture is not.
        args.append("--enforce-eager")
    return f"{' '.join(env)} \\\n    vllm serve {model} \\\n    {' '.join(args)}"


def require_server(model: str, worker: str, *, graph: bool = True,
                   max_model_len: int = 2048, tp: int = 1) -> str:
    """Return the base URL, or explain how to start the server and exit."""
    url = base_url()
    try:
        with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=5) as r:
            served = {m.get("id") for m in json.loads(r.read()).get("data", [])}
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[mia] no server at {url} ({e}).\n\nStart one in another terminal:\n\n"
              f"{serve_command(model, worker, graph=graph, max_model_len=max_model_len, tp=tp)}\n",
              file=sys.stderr)
        raise SystemExit(1)

    if served and model not in served:
        print(f"[mia] the server at {url} serves {sorted(served)}, not {model!r}. Either "
              f"restart it for this model or set MIA_DEMO_MODEL to one it serves.",
              file=sys.stderr)
        raise SystemExit(1)

    print(f"[mia] server at {url}, model {model}, MIA_WORKER={worker}"
          f"{' (FULL CUDA graphs)' if graph else ''}")
    return url


def print_evidence(elapsed_s: float, n_tokens: int, label: str = "") -> None:
    """Client-side timing. The capture counters live in the server process, not here."""
    tag = f"[evidence:{label}]" if label else "[evidence]"
    per_tok = (elapsed_s * 1000 / n_tokens) if n_tokens else float("nan")
    print(f"{tag} {elapsed_s * 1000:.1f} ms for {n_tokens} tokens "
          f"({per_tok:.2f} ms/token, measured at the client)")
    print(f"{tag} hook and aperture counters are server-side; start the server with "
          f"MIA_PROFILE=1 and read them from its log.")


def chat(prompt: str) -> list:
    """A one-turn chat message list, which is what the server endpoint takes."""
    return [{"role": "user", "content": prompt}]


def completion_text(response) -> str:
    return response.choices[0].message.content or ""


def completion_tokens(response) -> int:
    usage = getattr(response, "usage", None)
    return int(getattr(usage, "completion_tokens", 0) or 0)


def _port() -> str:
    url = base_url()
    tail = url.rstrip("/").rsplit(":", 1)[-1]
    return tail.split("/")[0] if tail[:1].isdigit() else "8770"


__all__ = ["HS", "QK", "STEER", "base_url", "serve_command", "require_server",
           "print_evidence", "chat", "completion_text", "completion_tokens"]
